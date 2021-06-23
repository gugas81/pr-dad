import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fire


class ConvNetMnistClasif(nn.Module):
    def __init__(self):
        super(ConvNetMnistClasif, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class FCNetMagnitudeMnistClasif(nn.Module):
    def __init__(self, use_bn: bool = False, use_dropout: bool = False, in_channels: int = 28**2):
        super(FCNetMagnitudeMnistClasif, self).__init__()
        self.use_dropout = use_dropout
        self.in_channels = in_channels

        self.fc1 = nn.Linear(self.in_channels, 512)

        self.bn1 = nn.BatchNorm1d(512)  # if use_bn else nn.Identity(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  # if use_bn else nn.Identity(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)  # if use_bn else nn.Identity(128)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)  # if use_bn else nn.Identity(64)

        self.fc5 = nn.Linear(64, 10)

        self.activ = F.relu

    def forward(self, x):
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

        x = self.fc5(x)
        return F.log_softmax(x)


class Trainer:
    def __init__(self, use_fft: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_epochs = 3
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 50
        self.use_fft = use_fft
        self.rand_perm = torch.randperm(28**2)
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

        self._init_models()

    def _init_models(self):
        self.conv_net_classif = ConvNetMnistClasif()
        self.conv_net_classif.to(device=self.device)
        self.optimizer_convnet = optim.SGD(self.conv_net_classif.parameters(),
                                           lr=self.learning_rate,
                                           momentum=self.momentum)

        self.fc_magnitude_classif = FCNetMagnitudeMnistClasif(in_channels=28**2)
        self.fc_magnitude_classif.to(device=self.device)
        self.optimizer_fc_magnitude = optim.SGD(self.fc_magnitude_classif.parameters(),
                                                lr=self.learning_rate,
                                                momentum=self.momentum)

    def prepare_data(self, data_batch: torch.Tensor):

        if self.use_fft:
            data_batch = torch.fft.fft2(data_batch, norm="ortho")
            data_batch = torch.fft.fftshift(data_batch, dim=(-2, -1))
            data_batch = torch.abs(data_batch)
        return data_batch

    def test(self, network: nn.Module):
        test_losses = []
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                data = self.prepare_data(data)
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        acc = correct / len(self.test_loader.dataset)
        print(f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {acc:.4%}')
        return test_losses

    def train_net(self, net_model: nn.Module, optimizer, epoch: int):
        train_losses = []
        train_counter = []
        net_model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            optimizer.zero_grad()
            data = self.prepare_data(data)
            output = net_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
        return train_losses, train_counter

    def run_train(self, net_model: nn.Module, optimizer):
        self.test(net_model)
        train_losses = []
        train_counter = []
        for epoch in range(1, self.n_epochs + 1):
            train_losses_batch, train_counter_batch = self.train_net(net_model, optimizer, epoch)
            train_losses += train_losses_batch
            train_counter += train_counter_batch
            test_losses_batch = self.test(net_model)
        return train_losses, train_counter, test_losses_batch

    def run_fc(self):
        train_losses, train_counter, test_losses_batch = self.run_train(self.fc_magnitude_classif,
                                                                        self.optimizer_fc_magnitude)
        return train_losses, train_counter, test_losses_batch

    def run_conv_magnitude(self):
        train_losses, train_counter, test_losses_batch = self.run_train(self.conv_net_classif, self.optimizer_convnet)
        return train_losses, train_counter, test_losses_batch


def example():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('module://backend_interagg')
    trainer_fft = Trainer(use_fft=True)
    train_losses_fc, train_counter_fc, test_losses_fc = trainer_fft.run_fc()
    train_losses_convnet, train_counter_convnet, test_losses_convnet = trainer_fft.run_conv_magnitude()

    plt.plot(test_losses_fc)
    plt.plot(test_losses_convnet)
    plt.legend(['test_losses_fc', 'test_losses_conv'])
    plt.show()

    plt.plot(train_losses_fc)
    plt.plot(train_losses_convnet)
    plt.legend(['train_losses_fc', 'train_losses_conv'])
    plt.show()
    # recon_net.run_train()



    # trainer = Trainer(use_fft=False)
    # trainer.run_fc()
    # trainer.run_conv_magnitude()


if __name__ == '__main__':
    fire.Fire()
