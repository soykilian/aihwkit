# Imports and Helper Functions
import os
from datetime import datetime
from IPython.display import display, clear_output
import ipywidgets as widgets

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from torch.optim import SGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig, UnitCellRPUConfig, InferenceRPUConfig, DigitalRankUpdateRPUConfig
from aihwkit.simulator.configs.devices import *
from aihwkit.simulator.configs.utils import PulseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.inference import BaseNoiseModel, PCMLikeNoiseModel, StateIndependentNoiseModel
from aihwkit.inference import GlobalDriftCompensation
from tqdm import tqdm

from aihwkit.simulator.configs import (
    RPUDataType,
    InferenceRPUConfig,
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference import ReRamCMONoiseModel
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.parameters.io import IOParametersIRDropT
USE_CUDA = 0
if torch.cuda.is_available():
    USE_CUDA = 1
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
plt.style.use('ggplot')

# Path to store datasets
PATH_DATASET = os.path.join('data', 'DATASET')

# Training parameters
SEED = 1
N_EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_CLASSES = 10
torch.manual_seed(SEED)

def load_images(bs):
    """Load images for train from torchvision datasets.
    
    Args:
        bs (int): batchsize
    """

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False)

    return train_data, validation_data

def create_digital_optimizer(model, learning_rate):
    """Create the conventional "digital" optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        nn.Module: SGD optimizer
    """
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    return optimizer

def create_analog_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        nn.Module: Analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
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
        nn.Module, nn.Module, float:  model, optimizer and loss for per epoch
    """
    total_loss = 0

    model.train()

    for images, labels in train_data:
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
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss


def test_evaluation(validation_data, model, criterion):
    """Test trained network.

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float:  model, loss, error, and accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok/total_images*100
        error = (1-predicted_ok/total_images)*100

    epoch_loss = total_loss / len(validation_data.dataset)

    return model, epoch_loss, error, accuracy


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, fig, print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    Returns:
        nn.Module, Optimizer, Tuple: model, optimizer,
            and a tuple of train losses, validation losses, and test
            error
    """
    train_losses = []
    valid_losses = []
    test_error = []

    # Train model
    for epoch in tqdm(range(0, epochs)):
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)

        # Validate_step
        with torch.no_grad():
            model, valid_loss, error, accuracy = test_evaluation(
                validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)
            
        plt.clf()
        plt.gca().set_prop_cycle(None)
        plt.plot(range(1, epoch+2), train_losses, marker="o", label="Training")
        plt.plot(range(1, epoch+2), valid_losses, marker="o", label="Validation")
        plt.gca().set_prop_cycle(None)
        plt.plot(epoch+1, train_losses[-1], marker="o", markersize=10)
        plt.plot(epoch+1, valid_losses[-1], marker="o", markersize=10)
        plt.xlim([0.5, epochs+0.5])
        plt.ylim([0, max(train_losses)+0.25])
        plt.xticks(range(1, epochs+2))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fig.canvas.draw()

    return model, optimizer, (train_losses, valid_losses, test_error)

def create_analog_network(rpu_config):
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
        nn.LogSoftmax(dim=1)
    )

    return model

def gen_rpu_config():
    input_prec = 6
    output_prec = 8
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True # do the bias of the MVM digitally
    my_rpu_config.mapping.max_input_size = 256
    my_rpu_config.mapping.max_output_size = 256

    #my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    my_rpu_config.noise_model = ReRamCMONoiseModel(g_max=88.19, g_min=9.0,
                                                acceptance_range=0.2)
    my_rpu_config.drift_compensation = None # by default is GlobalCompensation from PCM

    #my_rpu_config.drift_compensation = None
    my_rpu_config.forward = IOParametersIRDropT()
    my_rpu_config.forward.inp_res = 1 / (2**input_prec - 2)
    my_rpu_config.forward.out_res = 1 / (2**output_prec - 2)
    my_rpu_config.forward.is_perfect = False
    my_rpu_config.forward.out_noise = 0.06 # Output on the current addition (?)
    my_rpu_config.forward.ir_drop_g_ratio = 1.0 / 0.35 / 88e-6 # change to 25w-6 when using PCM
    my_rpu_config.forward.ir_drop = 1.0 # TODO set to 1.0 when activating IR drop effects
    my_rpu_config.forward.ir_drop_rs = 0.35 # Default: 0.15
    #my_rpu_config.forward.noise_management = NoiseManagementType.NONE # Rescale back the output with the scaling for normalizing the input
    my_rpu_config.forward.bound_management = BoundManagementType.NONE # No learning of the ranges
    my_rpu_config.forward.out_bound = 10.0  # quite restrictive
    return my_rpu_config

def main():
    #PRETRAINED_MODEL_PATH = "/Users/mvc/aihwkit/trained-LeNet5.pth"
    PRETRAINED_MODEL_PATH = "/u/mvc/aihwkit/trained-LeNet5.pth"
    batchsize = 64
    train_data, validation_data = load_images(batchsize)

    # Load the pretrained model
    dmodel = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
    rpu_config = gen_rpu_config()
    #dmodel = create_analog_network(rpu_config=rpu_config)
    #dmodel.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    
    criterion = nn.CrossEntropyLoss()
    dmodel, _, _, digital_accuracy = test_evaluation(validation_data, dmodel, criterion)
    print(f"Accuracy of the digital model: {digital_accuracy:.2f}%")
    amodel = convert_to_analog(dmodel, rpu_config)

    amodel.eval()
    amodel.program_analog_weights(noise_model=rpu_config.noise_model)
    amodel, _, _, accuracy = test_evaluation(validation_data, amodel, criterion)
    print(f"Accuracy of the analog model instantly after programming: {accuracy:.2f}%")
    print(f"Accuracy degradation: {digital_accuracy-accuracy:.2f}%")
    n_rep = 5
    exit
    t_inferences = [0, 1, 3600, 3600 * 24, 3600 * 24 * 365 * 10]
    #t_inferences = [10, 3600*24, 3600*24*365]
    drifted_test_accs = torch.zeros(size=(len(t_inferences),n_rep))
    for i,t in enumerate(t_inferences):
        for j in range(n_rep):
            amodel.drift_analog_weights(t)
            print("Drifted at t: ", t)
            amodel,_,_,accuracy = test_evaluation(validation_data, amodel, criterion)
            drifted_test_accs[i] += accuracy
            print(f"Accuracy of the analog model: {accuracy:.2f}%")
        drifted_test_accs[i] /= n_rep
    #torch.save(drifted_test_accs, "results/lenet_reram_baseline.pth")
    # TODO: average along dim =1 and save results over time in a numpy file
    torch.save(drifted_test_accs, "./lenet_baseline_reram.th")

if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()