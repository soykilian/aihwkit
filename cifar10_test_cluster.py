import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
import torch
import numpy as np
from tqdm import tqdm
import os

# - AIHWKIT related imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.utils import IOParameters

from aihwkit.simulator.parameters.io import IOParametersIRDropT
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference import ReRamCMONoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, n_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = torch.nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = torch.nn.Linear(64, n_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(n_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], n_classes=n_classes)


class TorchCutout(object):
    def __init__(self, length, fill=(0.0, 0.0, 0.0)):
        self.length = length
        self.fill = torch.tensor(fill).reshape(shape=(3, 1, 1))

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        img[:, y1:y2, x1:x2] = self.fill
        return img


# Load dataset
def load_cifar10(batch_size, path):
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            TorchCutout(length=8),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return trainloader, testloader


def gen_rpu_config():
    input_prec = 6
    output_prec = 8
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True # do the bias of the MVM digitally
    my_rpu_config.mapping.max_input_size = 256
    my_rpu_config.mapping.max_output_size = 256

    #my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    my_rpu_config.noise_model = ReRamCMONoiseModel(g_max=88.19, g_min=9.0,
                                                acceptance_range=2.0)
    my_rpu_config.drift_compensation = None # by default is GlobalCompensation from PCM

    #my_rpu_config.drift_compensation = None
    my_rpu_config.forward = IOParametersIRDropT()
    my_rpu_config.forward.inp_res = 1 / (2**input_prec - 2)
    my_rpu_config.forward.out_res = 1 / (2**output_prec - 2)
    my_rpu_config.forward.is_perfect = False
    #my_rpu_config.forward.out_noise = 0.0 # Output on the current addition (?)
    my_rpu_config.forward.ir_drop_g_ratio = 1.0 / 0.35 / 88e-6 # change to 25w-6 when using PCM
    my_rpu_config.forward.ir_drop = 1.0 # TODO set to 1.0 when activating IR drop effects
    my_rpu_config.forward.ir_drop_rs = 0.35 # Default: 0.15
    my_rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX # Rescale back the output with the scaling for normalizing the input
    my_rpu_config.forward.bound_management = BoundManagementType.NONE # No learning of the ranges
    my_rpu_config.forward.out_bound = 10.0  # quite restrictive
    return my_rpu_config

def train_step(model, optimizer, criterion, trainloader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, 100.0 * correct / total


def test_step(model, criterion, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"Test loss {test_loss/total:.4f} test acc. {100.*correct/total:.2f}%")
    return 100.0 * correct / total

if __name__=="__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    # - Get the dataloader
    batch_size = 128
    trainloader, testloader = load_cifar10(
        batch_size=batch_size, path=os.path.expanduser("~/Data/")
    )

    # - Change to True if one of the models should be re-trained
    retrain_finetuned_model = True

    lr = 0.05
    epochs = 200
    epochs_finetuning = 30

    PRETRAINED_MODEL_PATH =  "/u/mvc/aihwkit/Models/pre_trained_model.th"
    # - Define model, criterion, optimizer and scheduler.
    model = resnet32()
    model = model.to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    criterion = torch.nn.CrossEntropyLoss()
    rpu_config = gen_rpu_config()
    analog_model = convert_to_analog(model, rpu_config=rpu_config)
    if retrain_finetuned_model:
        optimizer = AnalogSGD(
            analog_model.parameters(), lr=lr / 10.0, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        test_accs = torch.empty(epochs_finetuning)
        train_losses =  torch.empty(epochs)
        train_accs =  torch.empty(epochs)
        test_acc =  torch.empty(epochs)

        for epoch in range(epochs_finetuning):
            train_loss, train_acc = train_step(analog_model, optimizer, criterion, trainloader)
            train_losses[epoch] = train_loss
            train_accs[epoch] = train_acc
            print(f"Epoch {epoch} Train loss: {train_loss:.4f} train acc. {train_acc:.2f}%")
            test_accs[epoch] = test_step(analog_model, criterion, testloader)
            scheduler.step()

        torch.save(analog_model.state_dict(), "Models/reram_finetuned_resnet.th")
        torch.save(test_accs, "Models/resnet_hwa_test_accs.th")
        torch.save(train_accs, "Models/resnet_hwa_train_accs.th")
        torch.save(train_losses, "Models/resnet_hwa_train_losses.th")

    rpu_conf = gen_rpu_config()
    analog_model = analog_model.eval()
    prog = test_step(analog_model, criterion, testloader)
    n_rep = 5
    t_inferences = [0, 1, 3600, 3600 * 24, 3600 * 24 * 365 * 10]
    drifted_test_accs = torch.zeros(size=(len(t_inferences),n_rep))
    for i,t in enumerate(t_inferences):
        for j in range(n_rep):
            analog_model.drift_analog_weights(t)
            print("Drifted at t: ", t)
            _,_,accuracy = test_step(analog_model, criterion, testloader)
            drifted_test_accs[i, j] = accuracy
            print(f"Accuracy of the analog model: {accuracy:.2f}%")
    torch.save(drifted_test_accs, "Models/resnet_drift_assessment.pth")
