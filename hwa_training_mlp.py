import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# torch imports
from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten, CrossEntropyLoss
from torch import device, no_grad, cuda
import torch
import torch.nn as nn
from torch import argmax
from torch.nn.functional import nll_loss
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
#imports from torchvision
from torchvision import datasets, transforms
from collections import OrderedDict
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.inference.noise.config import SimulationContextWrapper
from tqdm import tqdm

# aihwkit imports
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference.noise.reram import ReRamCMONoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.parameters.io import IOParametersIRDropT
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)

# DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DEVICE = device('cuda')
print('Running the simulation on: ', DEVICE)
train = False
PCM = False

errors = {}
enable_hook = False
d_outputs = {}
d_out_list = []
a_outputs = {}
a_out_list = []
weight_matrix  = {}
conductance_matrix = {}
image_counter = 0
layer_counter = 0


# Nome del file CSV dove salvare i risultati
csv_filename = "MLP_ErrorsVSLayers_PCM.csv"


def hook_fn(module, input, output):
    """Hook per calcolare l'errore MVM su ogni layer."""
    global enable_hook, image_counter, d_outputs, a_outputs, layer_counter # Dichiara che vuoi modificare la variabile globale

    layer_name = module.__class__.__name__  # Get the layer name
    # key = (image_counter, layer_name)
    if not enable_hook:
        d_outputs[module] = output
    else:
        a_outputs[module] = output

def create_analog_network(rpu_config):
    
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                        rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        Tanh(),
        Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        Tanh(),
        AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
        LogSoftmax(dim=1)
    )

    return model

def create_analog_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained

    Returns:
        Optimizer: created analog optimizer
    """
    
    optimizer = AnalogSGD(model.parameters(), lr=0.01) # we will use a learning rate of 0.01 as in the paper
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        train_dataset_loss: epoch loss of the train dataset
    """
    total_loss = 0
    correct = 0
    model.train()

    for images, labels in tqdm(train_data):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(labels).sum().item()
    train_dataset_loss = total_loss / len(train_data.dataset), 100*correct/len(train_data.dataset)

    return train_dataset_loss


def test_step(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns: 
        test_dataset_loss: epoch loss of the train_dataset
        test_dataset_error: error of the test dataset
        test_dataset_accuracy: accuracy of the test dataset
    """
    global enable_hook, image_counter, layer_counter, d_outputs, d_out_list, a_outputs, a_out_list

    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    image_counter = 0
    first_image = 1

    for images, labels in validation_data:
        images = images.to(DEVICE)
        images.reshape(images.shape[0], -1)
        labels = labels.to(DEVICE)

        # if first_image == 1:

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        test_dataset_accuracy = predicted_ok/total_images*100
        test_dataset_error = (1-predicted_ok/total_images)*100
        image_counter += 1
        first_image = 0
    test_dataset_loss = total_loss / len(validation_data.dataset)
    
    return test_dataset_loss, test_dataset_error, test_dataset_accuracy


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs=15, print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    """
    train_losses =  torch.empty(epochs)
    train_accs =  torch.empty(epochs)
    valid_losses =  torch.empty(epochs)
    test_acc =  torch.empty(epochs)

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        print("Epoch: ", epoch)
        train_loss, train_acc = train_step(train_data, model, criterion, optimizer)
        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        if epoch % print_every == (print_every - 1):
            # Validate_step
            with no_grad():
                valid_loss, error, accuracy = test_step(validation_data, model, criterion)
                valid_losses[epoch] = valid_loss
                test_acc[epoch] = accuracy

            print(f'Valid loss: {valid_loss:.4f}\t'
                  f'Test accuracy: {accuracy:.2f}%\t')
    torch.save(train_losses, "Models/resnet_hwa_train_loss.pth")
    torch.save(train_accs, "Models/resnet_hwa_train_acc.pth")
    torch.save(valid_losses, "Models/resnet_hwa_valid_loss.pth")
    torch.save(test_acc, "Models/resnet_hwa_test_acc.pth")
            

class LitAnalogModel(pl.LightningModule):
    def __init__(self, model, rpu_config, lr=0.05):
        super().__init__()

        # We simply convert the given model to analog on-the-fly
        self.analog_model = convert_to_analog_mapped(model, rpu_config)
        self.lr = lr

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        return self.analog_model(x_reshaped)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = nll_loss(logits, y)
        preds = argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        # we need to use the analog-aware optimizers
        optimizer = AnalogSGD(
            self.analog_model.parameters(),
            lr=self.lr,
        )
        return optimizer

PATH_DATASET = os.path.join('data', 'DATASET')
os.makedirs(PATH_DATASET, exist_ok=True)

def load_images():
    """Load images for train from torchvision datasets."""

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    test_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

    return train_data, test_data

def gen_rpu_config(noise_model: ReRamCMONoiseModel = None):
    input_prec = 6
    output_prec = 8
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True # do the bias of the MVM digitally
    my_rpu_config.mapping.max_input_size = 256
    my_rpu_config.mapping.max_output_size = 256
    my_rpu_config.forward = IOParametersIRDropT()
    if PCM:
        my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
        my_rpu_config.drift_compensation = GlobalDriftCompensation()
        my_rpu_config.forward.ir_drop_g_ratio = 1.0 / 0.35 / 25e-6 # change to 25w-6 when using PCM
    else:
        my_rpu_config.noise_model = noise_model
        my_rpu_config.drift_compensation = None # by default is GlobalCompensation from PCM
        my_rpu_config.forward.ir_drop_g_ratio = 1.0 / 0.35 / (noise_model.g_max*1e-6) # change to 25w-6 when using PCM

    #my_rpu_config.drift_compensation = None
    my_rpu_config.modifier.std_dev = 0.06
    my_rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    
    my_rpu_config.forward.inp_res = 1 / (2**input_prec - 2)
    my_rpu_config.forward.out_res = 1 / (2**output_prec - 2)
    my_rpu_config.forward.is_perfect = False
    #my_rpu_config.forward.out_noise = 0.0 # Output on the current addition (?)
    my_rpu_config.forward.ir_drop = 1.0 # TODO set to 1.0 when activating IR drop effects
    my_rpu_config.forward.ir_drop_rs = 0.35 # Default: 0.15
    my_rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX 
    my_rpu_config.forward.bound_management = BoundManagementType.NONE # No learning of the ranges
    my_rpu_config.forward.out_bound = 30.0  #10 quite restrictive
    return my_rpu_config

def create_digital_network():
    channel = [16, 32, 512, 128]
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=channel[2], out_features=channel[3]),
        nn.ReLU(),
        nn.Linear(in_features=channel[3], out_features=10),
        nn.LogSoftmax(dim=1)
    )

    return model

def main():

    torch.manual_seed(1)
    global enable_hook, d_outputs, a_outputs, d_out_list, a_out_list


    train_data, test_data = load_images()
    criterion = CrossEntropyLoss()
    #MODEL_PATH = "Models/finetuned_analog_resnet.th"
    MODEL_PATH = "/u/mvc/aihwkit/fp-mlp-mnist.pth" 

    model = torch.load(MODEL_PATH, map_location=DEVICE)
    # reram_model = convert_to_analog(model, rpu_config=rpu_config)
    # reram_model.eval()
    # analog_model = create_analog_network(rpu_config=rpu_config).to(DEVICE)
    # analog_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # print('model loaded')
    g_max = 90
    g_min = 10
    prog_overshoot =1.235
    single_device = True
    acceptance_range = 0.2
    reram_noise =ReRamCMONoiseModel(g_max=g_max, g_min=g_min,
                                                        acceptance_range=acceptance_range,
                                                        resistor_compensator=prog_overshoot,
                                                        single_device=single_device)
    rpu_config = gen_rpu_config(reram_noise)
    if train: 
        #create the modelpyth
        analog_model = create_analog_network(rpu_config).to(DEVICE)

        #define the analog optimizer
        optimizer = create_analog_optimizer(analog_model)

        training_loop(analog_model, criterion, optimizer, train_data, test_data)
        torch.save(analog_model.state_dict(), MODEL_PATH)
    else:
        dmodel = torch.load(MODEL_PATH, map_location="cuda")
        dmodel.eval()
        #_, _, digital_accuracy = test_step(test_data, dmodel, criterion)
        #print(f"Accuracy of the digital model: {digital_accuracy:.2f}%")
        dig_outputs = []
        all_outputs_per_layer = [[] for _ in range(len(dmodel.analog_model))]
        """
        with torch.no_grad():
            for batch in test_data:
                # If batch is a tuple (data, label), unpack it
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                for i, layer in enumerate(model.analog_model):
                    if i ==0:
                        x= torch.flatten(x, start_dim=2)
                    x = layer(x)
                    all_outputs_per_layer[i].append(x.clone())  # store batch output

        final_outputs_per_layer = [torch.cat(layer_outputs, dim=0) for layer_outputs in all_outputs_per_layer]
        """
        print(model)
        analog_model = convert_to_analog(model, rpu_config=rpu_config)
        
        analog_model.eval()
        analog_model.program_analog_weights(noise_model=rpu_config.noise_model)
        n_rep = 5
        #t_inferences = [10*60, 3600, 3600 * 24, 3600 * 24 * 365 * 10]

        #t_inferences = [0, 1, 3600, 3600*24, 3600 * 24 * 365 * 10]
        t_inferences = np.logspace(1, 8, 12)
        time_labels = {
            0: "0 s",
            1: "1 s", 
            3600: "1 h", 
            3600*24: "1 d", 
            3600*24*365*10: "10 y"
        }
        layers_print = ["AnalogLinear (1)", "AnalogLinear (2)", "AnalogLinear (3)", "LogSoftMax"]


        drifted_test_accs = torch.zeros((len(t_inferences)))
        stds = torch.zeros((len(t_inferences)))
        #torch.save(drifted_test_accs, "drift_compensation_NNs/MLP_dig_drift_compensation.th")
        _,_,accuracy = test_step(test_data, analog_model, criterion)
        for i,t in enumerate(t_inferences):
            #SimulationContextWrapper.t_inference = t
            accs = torch.zeros((n_rep))
            for n in range(n_rep):

                analog_model.drift_analog_weights(t)
                analog_outputs_per_layer = [[] for _ in range(len(dmodel.analog_model))]
                """
                with torch.no_grad():
                    for batch in test_data:
                        # If batch is a tuple (data, label), unpack it
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        for i, layer in enumerate(analog_model.analog_model):
                            if i ==0:
                                x= torch.flatten(x, start_dim=2)
                            x = layer(x)
                            analog_outputs_per_layer[i].append(x.clone()) 
                analog_final_outputs_per_layer = [torch.cat(layer_outputs, dim=0) for layer_outputs in analog_outputs_per_layer]
                mvm_error = []
                for i in range(len(analog_final_outputs_per_layer)-1):
                    error = torch.norm(analog_final_outputs_per_layer[i] - final_outputs_per_layer[i], p=2).item()/torch.norm(final_outputs_per_layer[i], p=2).item()
                    mvm_error.append(error)
                """
                #print("MVMe at t:",t, mvm_error)
                
                print("Drifted at t: ", t)
                _,_,accuracy = test_step(test_data, analog_model, criterion)
                accs[n] = accuracy
            print(accs, accs.std())
            drifted_test_accs[i] =accs.mean()
            stds[i] = accs.std()
        torch.save(drifted_test_accs, "drift_compensation_NNs/MLP_prog_overshoot_123.th")
        torch.save(drifted_test_accs, "drift_compensation_NNs/MLP_std_prog_overshoot_123.th")
        # Scrive l'intestazione solo se il file non esiste

 # THIS VERSION IS WITH Gtarget + 1.23 overshooting progamming approach
if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
