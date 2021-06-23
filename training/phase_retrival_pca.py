import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fire
from sklearn.decomposition import PCA


class FcPhaseRetrival(nn.Module):
    def __init__(self, use_bn: bool = False, use_dropout: bool = False, in_channels: int = 28 **2, out_channels: int = 64):
        super(FcPhaseRetrival, self).__init__()
        self.use_dropout = use_dropout
        self.in_channels = in_channels

        self.fc1 = nn.Linear(self.in_channels, self.in_channels // 2)

        self.bn1 = nn.BatchNorm1d(self.in_channels // 2)  # if use_bn else nn.Identity(512)

        self.fc2 = nn.Linear(self.in_channels // 2, self.in_channels //4)
        self.bn2 = nn.BatchNorm1d(self.in_channels)  # if use_bn else nn.Identity(256)

        self.fc3 = nn.Linear(self.in_channels // 4, out_channels //8)
        self.bn3 = nn.BatchNorm1d(out_channels)  # if use_bn else nn.Identity(128)

        self.fc4 = nn.Linear(out_channels // 8, out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)  # if use_bn else nn.Identity(64)

        self.activ = nn.LeakyReLU()

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.in_channels)
        x = self.fc1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        #         x = self.bn2(x)
        x = self.activ(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc3(x)
        #         x = self.bn3(x)
        x = self.activ(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc4(x)
        #         x = self.bn4(x)
        x = self.activ(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)

        x = x.view(-1, 1, 28, 28, 2)
        x = torch.view_as_complex(x)

        return x


class ReconPCAMnist:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_epochs = 2
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 50
        self.size_image =28

        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_test, shuffle=True)

        train_images, _ = next(iter(self.train_loader))  # Ignore train labels
        train_images = train_images.numpy()

        ## Fit PCA on training data
        self.pca = PCA(n_components=2, whiten=True)
        self.pca.fit(train_images.reshape([-1, self.size_imag **2]))

        self.fc_recon = FcPhaseRetrival()
        self.fc_recon.to(device=self.device)
        self.optimizer_fc_recon = optim.SGD(self.fc_recon.parameters(),
                                            lr=self.learning_rate,
                                            momentum=self.momentum)

    def train_fc_recon(self, epoch: int):
        train_losses = []
        train_counter = []
        self.fc_recon.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer_fc_recon.zero_grad()
            data = self.prepare_data(data)
            output_fft = self.fc_recon(data)

            loss = self.l2_loss(output_fft.abs(), data)
            loss.backward()
            self.optimizer_fc_recon.step()
            if batch_idx % self.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
        return train_losses, train_counter