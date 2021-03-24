import os

import torch
import torch.nn.functional as F
import torch.nn.modules as nn
# shape = (8, 192)
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from find_weights import get_weights, get_data

class CVAE(torch.nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        # 12288
        # encode
        in_channels = 457
        second_channels = 1024
        third_channels = 512
        fourth_channels = 128
        latent_channels = 1

        self.in_conv1 = nn.Conv1d(457, 1024, 3)
        self.in_conv2 = nn.Conv1d(457, 1024, 3)
        self.in_mu = nn.Conv1d(1024, 1, 3)
        self.in_logvar = nn.Conv1d(1024, 1, 3)

        # decode
        self.out_decode_conv2 = nn.ConvTranspose1d(1, 1024, 3)
        self.out_conv1 = nn.ConvTranspose1d(1024, 457, 3)
        self.out_conv2 = nn.ConvTranspose1d(1024, 457, 3)

        self.eps = None
        self.noise_fixed = True

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        x = self.in_conv1(x)
        x = F.relu(x)

        mu = self.in_mu(x)
        logvar = self.in_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.noise_fixed:
            if self.eps is None:
                self.eps = torch.randn((1, *std.size()[1:])).to("cuda:0")

            return mu + self.eps * std
        else:
            eps = torch.randn((1, *std.size()[1:])).to("cuda:0")
            return mu + eps * std

    def decode(self, z):
        x = self.out_decode_conv2(z)
        x = F.relu(x)
        x = self.out_conv1(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def loss_function(recon_x, x, mu, logvar, mask=None):
    # BCE = F.binary_cross_entropy(recon_x.view, x.view(-1, 784), reduction='sum')
    # MSE = F.mse_loss(torch.masked_select(recon_x, mask), torch.masked_select(x, mask)) # NOT BINARY CROSSENTROPY!
    MSE = F.l1_loss(torch.masked_select(recon_x, mask), torch.masked_select(x, mask))  # NOT BINARY CROSSENTROPY!

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, KLD

from collections import OrderedDict

class PolicyWrapper:
    def __init__(self, params: OrderedDict):
        self.names = [n for n in params.keys()]
        self.sizes = [t.size() for n, t in params.items()]
        self.widths = [t.size(-1) for n, t in params.items()]
        self.max_width = max(self.widths)
        self.masks = self.get_mask(params)

    def pad_right(self, params: OrderedDict):
        padded = []

        for n, t in params.items():
            if len(t.size()) == 2:
                p = torch.zeros(t.size(0), self.max_width)
                p[:, :t.size(1)] = t
            elif len(t.size()) == 1:
                p = torch.zeros(1, self.max_width)
                p[0, :t.size(0)] = t

            padded.append(p)

        return padded

    def get_mask(self, params: OrderedDict):
        mask = []

        for n, t in params.items():
            if len(t.size()) == 2:
                m = torch.zeros(t.size(0), self.max_width, dtype=torch.bool)
                m[:, :t.size(1)] = 1.0

            elif len(t.size()) == 1:
                m = torch.zeros(1, self.max_width, dtype=torch.bool)
                m[0, :t.size(0)] = 1.0

            mask.append(m)

        return mask

    def to_policy(self, padded_tensor):
        params = []

        r = 0
        for size, name in zip(self.sizes, self.names):
            if len(size) == 1:
                size = (1, size[0])

            dr, dc = size

            weights = padded_tensor[r:r+dr, :dc]
            params.append((name, weights.squeeze(dim=0)))
            r += dr

        return OrderedDict(params)



if __name__ == '__main__':
    torch.manual_seed(0)
    weight_file = '/home/joosep/PycharmProjects/StarCraft/latest/seed-scan/1614471632_5x5_eval_12x12_10A5P_fullmono_notime_noreset_epsilon_eval_seed_109/params/50_rnn_net_params.pkl'
    params = torch.load(weight_file)

    wrapper = PolicyWrapper(params)
    padded = torch.cat(wrapper.pad_right(params), dim=0)
    policy = wrapper.to_policy(padded)
    print(padded.size())

    for (no, original), (nr, reconstructed) in zip(params.items(), policy.items()):
        assert no == nr
        assert torch.sum(original - reconstructed) == 0, torch.sum(original - reconstructed)

    policies, targets = get_data()
    padded_weights = map(wrapper.pad_right, policies)

    data = torch.cat([torch.cat(w, dim=0).unsqueeze(dim=0) for w in padded_weights], dim=0)
    # data /= torch.max(data, dim=0)[0]
    # data[torch.isnan(data)] = 0.0

    targets = torch.cat([torch.from_numpy(t).unsqueeze(dim=0) for t in targets], dim=0)
    targets /= torch.max(targets, dim=0)[0]
    # targets[torch.isnan(targets)] = 0.0

    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=32)
    device = torch.device("cuda" if True else "cpu")

    mask = torch.cat(wrapper.masks, dim=0).to(device)

    if False and os.path.isfile('model.pt'):
        model = CVAE().to(device)
        weights = torch.load('model.pt', map_location=device)
        model.load_state_dict(weights)
        noise = torch.randn((1024, padded.size(0), 4)).to(device)
        x = model.decode(noise)
        best = torch.argmax(model.predict_performance(noise))
        test_policy = wrapper.to_policy(x[best].to("cpu"))
        torch.save(test_policy, "weights.pt")
        exit(0)

    model = CVAE().to(device)

    for name, module in model.named_modules():
        print(name, module)
        break

    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)

    # reconstruction warmup https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
    for epoch in range(50):
        train_loss = 0.0
        mse_loss = 0.0

        for batch_idx, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse

            loss.backward()
            train_loss += loss.item()
            mse_loss += mse.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}'.format(
                epoch,
                train_loss / len(train_loader),
                mse_loss / len(train_loader),
            )
        )

    # all together
    for epoch in range(50):
        train_loss = 0.0
        mse_loss = 0.0
        kld_loss = 0.0
        perf_loss =0.0

        for batch_idx, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse + kld

            ys = model.predict_performance(zs)
            perf_pred_loss = F.mse_loss(ys, target)
            loss += perf_pred_loss

            loss.backward()
            train_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            perf_loss += perf_pred_loss.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}, KLD Loss {:.6f}, Perf Pred Loss {:.6f}'.format(
                epoch,
                train_loss / len(train_loader),
                mse_loss / len(train_loader),
                kld_loss / len(train_loader),
                perf_loss / len(train_loader)
            )
        )


    model.noise_fixed = False
    # all together
    for epoch in range(200):
        train_loss = 0.0
        mse_loss = 0.0
        kld_loss = 0.0
        perf_loss =0.0

        for batch_idx, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse + kld

            ys = model.predict_performance(zs)
            perf_pred_loss = F.mse_loss(ys, target)
            loss += perf_pred_loss

            loss.backward()
            train_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            perf_loss += perf_pred_loss.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}, KLD Loss {:.6f}, Perf Pred Loss {:.6f}'.format(
                epoch,
                train_loss / len(train_loader),
                mse_loss / len(train_loader),
                kld_loss / len(train_loader),
                perf_loss / len(train_loader)
            )
        )
    torch.save(model.state_dict(), 'model.pt')
