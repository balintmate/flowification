from models.layers.Flatten import Flatten
from models.layers.Conv2d import Conv2d
from models.layers.activations import LeakyReLU, RqSpline
from models.layers.Linear import Linear
from models.layers.Dequantize import Dequantize
from models.inns import RQ_NSF, RQ_NSF_Conv


def getArchitecture(data, architecture):
    assert data in ["mnist", "cifar"]
    if data == "mnist":
        latent_dim = 8
        if architecture == 'mlp':
            layers = [
                Dequantize(),
                Flatten(),
                Linear(784, 512), RqSpline(),
                Linear(512, 256), RqSpline(),
                Linear(256, 128), RqSpline(),
                Linear(128, 64), RqSpline(),
                Linear(64, 32), RqSpline(),
                Linear(32, 8)
            ]
        if architecture == 'conv1':
            layers = [
                Dequantize(),
                Conv2d(1, 16, kernel_size=3, stride=2), RqSpline(),  # 14
                Conv2d(16, 24, kernel_size=2, stride=2), RqSpline(),  # 7
                Conv2d(24, 32, kernel_size=3, stride=2), RqSpline(),  # 3
                Conv2d(32, 48, kernel_size=2, stride=1), RqSpline(),  # 2
                Conv2d(48, 64, kernel_size=2, stride=1), RqSpline(),  # 1
                Flatten()]
            for _ in range(6):
                layers += [Linear(64, 64), RqSpline(), ]
            layers += [Linear(64, 32), RqSpline()]
            for _ in range(6):
                layers += [Linear(32, 32), RqSpline(), ]
            layers += [Linear(32, 8), RqSpline()]

        elif architecture == 'conv1_nsf':
            layers = [
                Dequantize(),
                Conv2d(1, 16, kernel_size=3, stride=2), RqSpline(),  # 14
                Conv2d(16, 24, kernel_size=2, stride=2), RqSpline(),  # 7
                Conv2d(24, 32, kernel_size=3, stride=2), RqSpline(),  # 3
                Conv2d(32, 48, kernel_size=2, stride=1), RqSpline(),  # 2
                Conv2d(48, 64, kernel_size=2, stride=1), RqSpline(),  # 1
                Flatten(),
                RQ_NSF(dim=64, nstack=2),
                Linear(64, 32), RqSpline(),
                RQ_NSF(dim=32, nstack=2),
                Linear(32, 8), RqSpline(),
                RQ_NSF(dim=8, nstack=2),
            ]

        if architecture == 'conv2':
            layers = [
                Dequantize(),
                Conv2d(1, 4, kernel_size=2, stride=2), RqSpline(),  # 14
                Conv2d(4, 16, kernel_size=2, stride=2), RqSpline(),  # 7
                Conv2d(16, 32, kernel_size=2, stride=2), RqSpline(),  # 4
                Conv2d(32, 48, kernel_size=2, stride=2), RqSpline(),  # 2
                Conv2d(48, 64, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten()]
            for _ in range(6):
                layers += [Linear(64, 64), RqSpline(), ]
            layers += [Linear(64, 32), RqSpline()]
            for _ in range(6):
                layers += [Linear(32, 32), RqSpline(), ]
            layers += [Linear(32, 8), RqSpline()]

        if architecture == 'conv2_nsf':
            layers = [
                Dequantize(),
                Conv2d(1, 4, kernel_size=2, stride=2), RqSpline(),  # 14
                RQ_NSF_Conv(channels=4, nstack=2),
                Conv2d(4, 16, kernel_size=2, stride=2), RqSpline(),  # 7
                RQ_NSF_Conv(channels=16, nstack=2),
                Conv2d(16, 24, kernel_size=2, stride=2), RqSpline(),  # 4
                RQ_NSF_Conv(channels=24, nstack=2),
                Conv2d(24, 32, kernel_size=2, stride=2), RqSpline(),  # 2
                RQ_NSF_Conv(channels=32, nstack=2),
                Conv2d(32, 64, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten(),
                RQ_NSF(dim=64, nstack=2),
                Linear(64, 32), RqSpline(),
                RQ_NSF(dim=32, nstack=2),
                Linear(32, 8), RqSpline(),
                RQ_NSF(dim=8, nstack=2),
            ]


    elif data == "cifar":
        latent_dim = 128
        if architecture == 'mlp':
            layers = [
                Dequantize(),
                Flatten(),
                Linear(3072, 1024), RqSpline(),
                Linear(1024, 512), RqSpline(),
                Linear(512, 512), RqSpline(),
                Linear(512, 256), RqSpline(),
                Linear(256, 128)
            ]

            latent_dim = 48
            layers = [
                Dequantize(),
                MakePatches(patch_size=4)]  # D1= 3*4*4, D2= 8*8
            for _ in range(5):
                layers += [MixerLayer(D1=48, D2=64, MLPlayers=4)]
            layers += [
                Fold(orig_size=(32, 32), patch_size=4),
                Conv2d(3, 6, kernel_size=4, stride=4),
                MakePatches(patch_size=2)
            ]  # (6,8,8)
            for _ in range(5):
                layers += [MixerLayer(D1=24, D2=16, MLPlayers=4)]

            layers += [
                Fold(orig_size=(8, 8), patch_size=2),
                Conv2d(6, 3, kernel_size=2, stride=2),  # (3,4,4)
                MakePatches(patch_size=2),  # D1= 3*2*2, D2= 2*2
            ]  # (6
            for _ in range(5):
                layers += [MixerLayer(D1=12, D2=4, MLPlayers=4)]
            layers += [Flatten()]
       
        if architecture == 'conv1':  # overlapping kernels
            layers = [
                Dequantize(),
                Conv2d(3, 27, kernel_size=3, stride=2), RqSpline(),  # 16
                Conv2d(27, 64, kernel_size=3, stride=2), RqSpline(),  # 8
                Conv2d(64, 100, kernel_size=3, stride=2), RqSpline(),  # 4
                Conv2d(100, 112, kernel_size=3, stride=2), RqSpline(),  # 2
                Conv2d(112, 128, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten()]
            for _ in range(10):
                layers += [Linear(128, 128), RqSpline()]

        if architecture == 'conv1_nsf':  # overlapping kernels + rq_nsf
            layers = [
                Dequantize(),
                Conv2d(3, 27, kernel_size=3, stride=2), RqSpline(),  # 16
                RQ_NSF_Conv(channels=27, nstack=4),
                Conv2d(27, 48, kernel_size=3, stride=2), RqSpline(),  # 8
                RQ_NSF_Conv(channels=48, nstack=4),
                Conv2d(48, 72, kernel_size=3, stride=2), RqSpline(),  # 4
                RQ_NSF_Conv(channels=72, nstack=4),
                Conv2d(72, 96, kernel_size=3, stride=2), RqSpline(),  # 2
                Conv2d(96, 128, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten()] + \
                [RQ_NSF(dim=128, nstack=8)]

        elif architecture == 'conv2':  # non-overlapping kernels
            layers = [
                Dequantize(),
                Conv2d(3, 12, kernel_size=2, stride=2), RqSpline(),  # 16
                Conv2d(12, 48, kernel_size=2, stride=2), RqSpline(),  # 8
                Conv2d(48, 100, kernel_size=2, stride=2), RqSpline(),  # 4
                Conv2d(100, 112, kernel_size=2, stride=2), RqSpline(),  # 2
                Conv2d(112, 128, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten()]
            for _ in range(12):
                layers += [Linear(128, 128), RqSpline()]

        elif architecture == 'conv2_nsf':  # non-overlapping kernels + rq_nsf


            layers = [
                Dequantize(),
                Conv2d(3, 12, kernel_size=2, stride=2), RqSpline(),  # 16
                RQ_NSF_Conv(channels=12, nstack=4),
                Conv2d(12, 48, kernel_size=2, stride=2), RqSpline(),  # 8
                RQ_NSF_Conv(channels=48, nstack=4),
                Conv2d(48, 72, kernel_size=2, stride=2), RqSpline(),  # 4
                RQ_NSF_Conv(channels=72, nstack=4),
                Conv2d(72, 96, kernel_size=2, stride=2), RqSpline(),  # 2
                Conv2d(96, 128, kernel_size=2, stride=2), RqSpline(),  # 1
                Flatten()]
            layers += [RQ_NSF(dim=128, nstack=6)]

    return layers, latent_dim