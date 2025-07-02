
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
import torch
import numpy as np
from tqdm import tqdm

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

input_prec = 6
output_prec = 8
wire = 0.35
PCM = False

def gen_rpu_config(noise_model, pcm= False):
    input_prec = 6
    output_prec = 8
    my_rpu_config = InferenceRPUConfig()
    
    my_rpu_config.mapping.digital_bias = True # do the bias of the MVM digitally
    my_rpu_config.mapping.max_input_size = 256
    my_rpu_config.mapping.max_output_size = 256
    my_rpu_config.forward = IOParametersIRDropT()
    if pcm:
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
    my_rpu_config.mapping.weight_scaling_omega = 1.0
    my_rpu_config.mapping.weight_scaling_columnwise = False
    my_rpu_config.mapping.out_scaling_columnwise = False
    my_rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC 
    my_rpu_config.forward.inp_res = 1 / (2**input_prec - 2)
    my_rpu_config.forward.out_res = 1 / (2**output_prec - 2)
    my_rpu_config.forward.is_perfect = False
    #my_rpu_config.forward.out_noise = 0.0 # Output on the current addition (?)
    my_rpu_config.forward.ir_drop = 1.0 # TODO set to 1.0 when activating IR drop effects
    my_rpu_config.forward.ir_drop_rs = 0.35 # Default: 0.15
    my_rpu_config.pre_post.input_range.enable = False
    
    #my_rpu_config.pre_post.input_range.manage_output_clipping = True
    my_rpu_config.pre_post.input_range.decay = 0.001
    my_rpu_config.pre_post.input_range.input_min_percentage = 0.95
    my_rpu_config.pre_post.input_range.output_min_percentage = 0.95
    my_rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX # Rescale back the output with the scaling for normalizing the input
    #my_rpu_config.forward.bound_management = BoundManagementType.ITERATIVE
    my_rpu_config.forward.out_bound = 20.0  # quite restric
    return my_rpu_config

def train_step(model, optimizer, criterion, trainloader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
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
def main():
    g_max = 90
    g_min = 10
    acceptance_range=0.2
    single_device=True
    prog_overshoot=0.0
    reram_noise =ReRamCMONoiseModel(g_max=g_max, g_min=g_min,
                                                            acceptance_range=acceptance_range,
                                                            resistor_compensator=prog_overshoot,
                                                                single_device=single_device)
    torch.manual_seed(0)
    np.random.seed(0)
    import os
    # - Get the dataloader
    batch_size = 128
    trainloader, testloader = load_cifar10(
        batch_size=batch_size, path=os.path.expanduser("~/Data/")
    )

    # - Change to True if one of the models should be re-trained
    retrain_baseline = False
    retrain_finetuned_model = True

    # - Some hyperparameters
    lr = 0.05
    epochs = 200
    epochs_finetuning = 100
    model = resnet32()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    analog_model = convert_to_analog(model, gen_rpu_config(noise_model=reram_noise))
    analog_model.load_state_dict(torch.load("/u/mvc/aihwkit/notebooks/tutorial/Models/hwa_reram_June_resnet.th", map_location=device))
    if retrain_finetuned_model:
        optimizer = AnalogSGD(
            analog_model.parameters(), lr=lr / 10.0, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        test_accs = torch.empty(epochs_finetuning)
        pbar = tqdm(range(epochs_finetuning))
        for epoch in pbar:
            train_loss, train_acc = train_step(analog_model, optimizer, criterion, trainloader)
            pbar.set_description(f"Epoch {epoch} Train loss: {train_loss:.4f} train acc. {train_acc:.2f}%")
            test_accs[epoch] = test_step(analog_model, criterion, testloader)
            scheduler.step()
            torch.save(analog_model.state_dict(), "/u/mvc/aihwkit/notebooks/tutorial/Models/hwa_reram_June_resnet_epochs.th")
        torch.save(analog_model.state_dict(), "/u/mvc/aihwkit/notebooks/tutorial/Models/hwa_reram_June_resnet_final.th")

    t_inferences = [1, 60*10, 3600, 3600 * 24, 3600 * 24*7, 3600 * 24 *30, 3600 * 24 *365, 3600 * 24 *365*2, 3600 * 24 *365*5, 3600 * 24 * 365 * 10]    
    print("Programming: ",test_step(analog_model, criterion, testloader))
    analog_model.eval()
    analog_model.program_analog_weights()
    labels = ["0s",  "10y"]
    color = [ 'lightskyblue', 'lightcoral']
    n_rep = 1
    drifted_test_accs = torch.zeros(size=(len(t_inferences), n_rep))
    #stds = torch.zeros(size=(len(t_inferences)))
    for i,t in enumerate(t_inferences):
        for j in range(n_rep):
            #SimulationContextWrapper.t_inference = t
            analog_model.drift_analog_weights(t)
            print("Drifted at t: ", t)
            accuracy = test_step(analog_model, criterion, testloader)
            drifted_test_accs[i,j] = accuracy 
    torch.save(drifted_test_accs, "resnet_hwa_baseline_1m_10y.th")

if __name__ == "__main__":
    # Execute only if run as the entry point into the program
    main()
