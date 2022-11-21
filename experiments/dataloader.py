import torch
import torchvision
from torch.utils.data import DataLoader



def Data(data, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data == "mnist":
        train_set = torchvision.datasets.MNIST(
            root="..",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_set = torchvision.datasets.MNIST(
            root="..",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        resolution = 28
        trainloader = DataLoader(train_set, batch_size=len(train_set))
        testloader = DataLoader(test_set, batch_size=len(test_set))
        X_train, y_train = next(iter(trainloader))
        X_test, y_test = next(iter(testloader))

        train_set = torch.utils.data.TensorDataset(
            X_train.to(device), y_train.to(device))
        test_set = torch.utils.data.TensorDataset(
            X_test.to(device), y_test.to(device))
        trainloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True)

    elif data == "cifar":
        train_set = torchvision.datasets.CIFAR10(
            root="..",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_set = torchvision.datasets.CIFAR10(
            root="..",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        num_pixels = 3072
        resolution = 32
        trainloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    elif data[:6] == "CelebA":  # kaggle datasets download -d lamsimon/celebahq
        resolution = int(data[6:])
        TRANSFORM = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((resolution, resolution)),
                torchvision.transforms.ToTensor(),
            ]
        )
        TRAIN_FOLDER = "../celeba_hq/train"
        TEST_FOLDER = "../celeba_hq/val"
        train_set = torchvision.datasets.ImageFolder(
            TRAIN_FOLDER, transform=TRANSFORM)
        test_set = torchvision.datasets.ImageFolder(
            TEST_FOLDER, transform=TRANSFORM)
        num_pixels = 3 * resolution * resolution
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                num_workers=4)

    elif data == "ImageNet64":  # kaggle datasets download -d wangzilin20078/imagenet64
        resolution = 64
        TRANSFORM = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((resolution, resolution)),
                torchvision.transforms.ToTensor(),
            ]
        )
        TRAIN_FOLDER = "../ImageNet64/ILSVRC/Data/CLS-LOC/train"
        TEST_FOLDER = "../ImageNet64/ILSVRC/Data/CLS-LOC/val"
        train_set = torchvision.datasets.ImageFolder(
            TRAIN_FOLDER, transform=TRANSFORM)
        test_set = torchvision.datasets.ImageFolder(
            TEST_FOLDER, transform=TRANSFORM)
        num_pixels = 3 * resolution * resolution
        trainloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        testloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return trainloader, testloader
