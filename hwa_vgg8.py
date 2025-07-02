# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 11: analog CNN.

SVHN dataset on Analog Network using weight scaling.

Learning rates of η = 0.1 for all the epochs with minibatch 128.
"""
# pylint: disable=invalid-name
from tqdm import tqdm
import os
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
import torch 
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.io import IOParametersIRDropT
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference import ReRamCMONoiseModel
from aihwkit.simulator.parameters.io import IOParametersIRDropT
# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join("data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VGG8")

SAVE_PATH = "/u/mvc/aihwkit/notebooks/tutorial/Models/pre-trained-vgg8.th"
SAVE_ANALOG_PATH = "/u/mvc/aihwkit/notebooks/tutorial/Models/hwa_2t2r_vgg8.th"
# Training parameters
SEED = 10
N_EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_CLASSES = 10
WEIGHT_SCALING_OMEGA = 0.6  # Should not be larger than max weight.

# Select the device model to use in the training. In this case we are using one of the preset,
# but it can be changed to a number of preset to explore possible different analog devices
mapping = MappingParameter(weight_scaling_omega=WEIGHT_SCALING_OMEGA)
RPU_CONFIG = GokmenVlasovPreset(mapping=mapping)
RPU_CONFIG.runtime.offload_gradient = True
RPU_CONFIG.runtime.offload_input = True


def load_images():
    """Load images for train from torchvision datasets."""
    mean = Tensor([0.4377, 0.4438, 0.4728])
    std = Tensor([0.1980, 0.2010, 0.1970])

    print(f"Normalization data: ({mean},{std})")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.SVHN(PATH_DATASET, download=True, split="train", transform=transform)
    val_set = datasets.SVHN(PATH_DATASET, download=True, split="test", transform=transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data


def create_digital_network():
    """Create a Vgg8 inspired analog model.

    Returns:
       nn.Module: VGG8 model
    """
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = torch.nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[0],
            out_channels=channel[0],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Conv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[1],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Conv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[2],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        nn.Linear(in_features=16 * channel[2], out_features=fc_size),
        nn.ReLU(),
        nn.Linear(in_features=fc_size, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    )
    return model


def create_analog_network():
    """Create a Vgg8 inspired analog model.

    Returns:
       nn.Module: VGG8 model
    """
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = AnalogSequential(
        AnalogConv(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[0],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[2],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        AnalogLinear(in_features=16 * channel[2], out_features=fc_size, rpu_config=RPU_CONFIG),
        nn.ReLU(),
        nn.Linear(in_features=fc_size, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    )
    return model


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
    Returns:
        Optimizer: optimizer
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
        nn.Module, Optimizer, float: model, optimizer, and epoch loss
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
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
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

        _, predicted = torch_max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    epoch_loss = total_loss / len(validation_data.dataset)

    return model, epoch_loss, error, accuracy


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, fp_baseline, print_every=1):
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
        nn.Module, Optimizer, Tuple: model, optimizer, and a tuple of
            lists of train losses, validation losses, and test error

    """
    train_losses = []
    valid_losses = []
    test_error = []
    acc_list = []
    decrease_counter = 0
    max_decreases = 5
    model_snapshots = {}

    # Train model
    optimizer = AnalogSGD(
    model.parameters(), lr=LEARNING_RATE / 10.0, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        pbar.set_description(f"Epoch {epoch} Train loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        if epoch % print_every == (print_every - 1):
            # Validate_step
            with no_grad():
                model, valid_loss, error, accuracy = test_evaluation(
                    validation_data, model, criterion
                )
                valid_losses.append(valid_loss)
                test_error.append(error)
                acc_list.append(accuracy)
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Test accuracy: {accuracy:.2f}%\t"
            )
            print(accuracy, fp_baseline)
            if abs(accuracy - fp_baseline) < 0.5:
                break
        model_snapshots[epoch] = copy.deepcopy(model.state_dict())
        scheduler.step()
        if len(acc_list) > 1:
            if accuracy < acc_list[-2]:
                decrease_counter += 1
            else:
                decrease_counter = 0  # Reset if it doesn't decrease

        # Stop if accuracy decreased for 5 consecutive checks
        if decrease_counter >= max_decreases:
            print(f"Stopping early at epoch {epoch} due to 5 consecutive accuracy drops.")

            # Restore model to 5 epochs ago (if possible)
            restore_epoch = epoch - (print_every * max_decreases)
            if restore_epoch in model_snapshots:
                model.load_state_dict(model_snapshots[restore_epoch])
                print(f"Model restored to epoch {restore_epoch}")
            else:
                print("Warning: no snapshot available for 5 epochs ago. Using latest best model.")
            break
    torch.save(model.state_dict(), SAVE_ANALOG_PATH)

    # Save results and plot figures

    return model, optimizer, (train_losses, valid_losses, test_error)


def plot_results(train_losses, valid_losses, test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, "r-s", valid_losses, "b-o")
    plt.title("aihwkit VGG8")
    plt.legend(fig[:2], ["Training Losses", "Validation Losses"])
    plt.xlabel("Epoch number")
    plt.ylabel("Loss [A.U.]")
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_losses.png"))
    plt.close()

    fig = plt.plot(test_error, "r-s")
    plt.title("aihwkit VGG8")
    plt.legend(fig[:1], ["Test Error"])
    plt.xlabel("Epoch number")
    plt.ylabel("Test Error [%]")
    plt.yscale("log")
    plt.ylim((5e-1, 1e2))
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_error.png"))
    plt.close()

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

def main():
    """Train a PyTorch CNN analog model with the MNIST dataset."""
    # Make sure the directory where to save the results exist.
    # Results include: Loss vs Epoch graph, Accuracy vs Epoch graph and vector data.
    os.makedirs(RESULTS, exist_ok=True)
    manual_seed(SEED)

    # Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model.
    model = create_digital_network().to(DEVICE)
    hwa_training = True
    criterion = nn.CrossEntropyLoss()
    if hwa_training:
        model.load_state_dict(torch.load(SAVE_PATH, map_location='cuda'))
        _,_, _, fp_baseline = test_evaluation(validation_data=validation_data, model=model, criterion=criterion)
        print("FP BASELINE ACC:", fp_baseline)
        g_max = 50
        g_min = 10
        prog_overshoot = 0.0
        single_device = False
        acceptance_range = 0.2
        reram_noise =ReRamCMONoiseModel(g_max=g_max, g_min=g_min,
                                                            acceptance_range=acceptance_range,
                                                            resistor_compensator=prog_overshoot,
                                                            single_device=single_device)
        rpu_config= gen_rpu_config(reram_noise)
        a_model = convert_to_analog(model, rpu_config=rpu_config).to(DEVICE)
        print(a_model)
        optimizer = create_sgd_optimizer(model, LEARNING_RATE)
        a_model, _, _ = training_loop(a_model, criterion, optimizer, train_data, validation_data, N_EPOCHS, fp_baseline)
    else:
        if USE_CUDA:
            a_model.cuda()
        print(a_model)

        print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started Vgg8 Example")

        optimizer = create_sgd_optimizer(model, LEARNING_RATE)


        model, optimizer, _ = training_loop(
            model, criterion, optimizer, train_data, validation_data, N_EPOCHS
        )

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed Vgg8 Example")


if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
