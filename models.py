import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, block):
        super(Residual, self).__init__()
        self.block = block
    def forward(self, x):
        return self.block(x) + x

class Generator(nn.Module):
    def __init__(self, n_res):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_res):
            model += [Residual(nn.Sequential(*[
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            ])),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        return self.model(x)