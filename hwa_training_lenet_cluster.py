import os

# torch imports
from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten, CrossEntropyLoss
from torch import device, cuda, no_grad
import torch
import torch.nn as nn

#imports from torchvision
from torchvision import datasets, transforms

from tqdm import tqdm

# aihwkit imports
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference import ReRamCMONoiseModel
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.parameters.io import IOParametersIRDropT
from aihwkit.inference import GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
print('Running the simulation on: ', DEVICE)
train = False
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
        test_dataset_accuracy = predicted_ok/total_images*100
        test_dataset_error = (1-predicted_ok/total_images)*100

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
    torch.save(train_losses, "Models/lenet_hwa_pcm_train_loss.pth")
    torch.save(train_accs, "Models/lenet_hwa_pcm_train_acc.pth")
    torch.save(valid_losses, "Models/lenet_hwa_pcm_valid_loss.pth")
    torch.save(test_acc, "Models/lenet_hwa_pcm_test_acc.pth")
            

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

def gen_rpu_config(noise_model=None):
    input_prec = 6
    output_prec = 8
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True # do the bias of the MVM digitally
    #my_rpu_config.mapping.max_input_size = 256
    #my_rpu_config.mapping.max_output_size = 256
    my_rpu_config.forward = IOParametersIRDropT()
    my_rpu_config.noise_model = noise_model
    my_rpu_config.drift_compensation = None    #my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    #my_rpu_config.drift_compensation = GlobalDriftCompensation()
    my_rpu_config.forward.ir_drop_g_ratio = 1.0 / 0.35 / (noise_model.g_max*1e-6) # change to 25w-6 when using PCM

    #my_rpu_config.drift_compensation = None
    my_rpu_config.modifier.std_dev = 0.06
    my_rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    
    my_rpu_config.forward.inp_res = 1 / (2**input_prec - 2)
    my_rpu_config.forward.out_res = 1 / (2**output_prec - 2)
    my_rpu_config.forward.is_perfect = True
    #my_rpu_config.forward.out_noise = 0.0 # Output on the current addition (?)
    my_rpu_config.forward.ir_drop = 1.0 # TODO set to 1.0 when activating IR drop effects
    my_rpu_config.forward.ir_drop_rs = 0.35 # Default: 0.15
    my_rpu_config.pre_post.input_range.enable = True
    
    #my_rpu_config.pre_post.input_range.manage_output_clipping = True
    my_rpu_config.pre_post.input_range.decay = 0.001
    my_rpu_config.pre_post.input_range.input_min_percentage = 0.95
    my_rpu_config.pre_post.input_range.output_min_percentage = 0.95
    #my_rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX # Rescale back the output with the scaling for normalizing the input
    my_rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    my_rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    my_rpu_config.clip.sigma = 2.5
    my_rpu_config.forward.out_bound = 10.0  # quite restric
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
    train_data, test_data = load_images()
    g_max = 90
    g_min = 10
    prog_overshoot =0.0
    single_device = True
    acceptance_range = 0.2
    reram_noise =ReRamCMONoiseModel(g_max=g_max, g_min=g_min,
                                                        acceptance_range=acceptance_range,
                                                        resistor_compensator=prog_overshoot,
                                                        single_device=single_device)
    rpu_config= gen_rpu_config(reram_noise)
    criterion = CrossEntropyLoss()
    if train: 
        #create the model
        analog_model = create_analog_network(rpu_config).to(DEVICE)

        #define the analog optimizer
        optimizer = create_analog_optimizer(analog_model)

        training_loop(analog_model, criterion, optimizer, train_data, test_data)
        torch.save(analog_model, "Models/lenet_pcm_hwa.pth")
    else:
        #model = create_digital_network()
        #model.load_state_dict(torch.load("Models/finetuned_analog_resnet.th"))
        MODEL_PATH = "lenet_reram_hwa.pth"
        analog_model = create_analog_network(rpu_config).to(DEVICE)
        #model = torch.load(MODEL_PATH, map_location="cpu")
        #_,_,accuracy = test_step(test_data, model, criterion)
        #print("FP acc", accuracy)
        #analog_model = convert_to_analog(model, rpu_config=rpu_config).to(DEVICE)
        #analog_model = create_analog_network(rpu_config=rpu_config).to(DEVICE)
        analog_model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"))
        analog_model.eval()
        #analog_model.program_analog_weights(noise_model=rpu_config.noise_model)
        n_rep = 1
        t_inferences = [0, 1, 3600, 3600 * 24, 3600 * 24 * 365 * 10]
        drifted_test_accs = torch.zeros(size=(len(t_inferences),n_rep))
        for i,t in enumerate(t_inferences):
            for j in range(n_rep):
                analog_model.drift_analog_weights(t)
                print("Drifted at t: ", t)
                _,_,accuracy = test_step(test_data, analog_model, criterion)
                drifted_test_accs[i, j] = accuracy
                print(f"Accuracy of the analog model: {accuracy:.2f}%")
        #torch.save(drifted_test_accs, "Models/resnet_drift_assessment.pth")
        # TODO: average along dim =1 and save results over time in a numpy file

if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
