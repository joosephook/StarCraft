import os

import torch
import torch.nn.functional as F
import torch.nn.modules as nn
# shape = (8, 192)
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from find_weights import get_weights, get_data

from numpy.random import default_rng

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class GRUVAE(torch.nn.Module):
    def __init__(self, device):
        super(GRUVAE, self).__init__()
        self.device = device
        # 12288
        # encode
        input_size = 192
        second_size = 64
        latent_size = 4

        self.gru1 = nn.GRU(input_size=input_size, hidden_size=second_size, batch_first=True, bidirectional=True)
        self.gru2mu = nn.GRU(input_size=second_size*2, hidden_size=latent_size, batch_first=True, bidirectional=True)
        self.gru2logvar = nn.GRU(input_size=second_size*2, hidden_size=latent_size, batch_first=True, bidirectional=True)

        # decode
        self.gru3 = nn.GRU(input_size=latent_size*2, hidden_size=second_size, batch_first=True, bidirectional=True)
        self.gru4 = nn.GRU(input_size=second_size*2, hidden_size=input_size, batch_first=True)


        # predict performance
        self.gru5 = nn.GRU(input_size=latent_size*2, hidden_size=second_size, batch_first=True, bidirectional=True)
        self.perf = nn.Linear(second_size*2, 1)
        self.eps = None
        self.noise_fixed = True

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        xs = x
        xs, _ = self.gru1(x)  # hidden defaults to 0 if not specified
        xs = F.tanh(xs)

        mu_hs, _ = self.gru2mu(xs)
        # mu_hs = torch.tanh(mu_hs)  # hidden defaults to 0
        logvar_hs, _ = self.gru2logvar(xs)
        # logvar_hs = torch.tanh(logvar_hs)

        # return mu_hs.view(-1), logvar_hs.view(-1)
        # return mu_hs, logvar_hs
        return mu_hs, logvar_hs

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        if self.noise_fixed:
            if self.eps is None:
                self.eps = torch.randn((1, *std.size()[1:])).to(self.device)#.to("cuda:0")


            return mu + self.eps * std
        else:
            eps = torch.randn((1, *std.size()[1:])).to(self.device)#("cuda:0")
            return mu + eps * std

    def decode(self, z):
        # hs = z.reshape(-1, 1000)
        hs = z
        hs, out = self.gru3(hs)
        hs = F.tanh(hs)
        hs, out = self.gru4(hs)
        # hs = torch.tanh(hs)

        return hs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def predict_performance(self, z):
        hiddens, h = self.gru5(z)
        h = torch.cat([h[0:1, :], h[1:2, :]], dim=-1).squeeze(dim=0)
        return self.perf(F.relu(h)).squeeze(dim=0)


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


def pad_right(tensors):
    max_width = max(t.size(-1) for t in tensors)
    padded = []

    for t in tensors:
        if len(t.size()) == 2:
            p = torch.zeros(t.size(0), max_width)
            p[:, :t.size(1)] = t
        elif len(t.size()) == 1:
            p = torch.zeros(1, max_width)
            p[0, :t.size(0)] = t

        padded.append(p)

    return padded


def get_mask(tensors):
    max_width = max(t.size(-1) for t in tensors)
    mask = []

    for t in tensors:
        if len(t.size()) == 2:
            m = torch.zeros(t.size(0), max_width, dtype=torch.bool)
            m[:, :t.size(1)] = 1.0

        elif len(t.size()) == 1:
            m = torch.zeros(1, max_width, dtype=torch.bool)
            m[0, :t.size(0)] = 1.0

        mask.append(m)

    return mask


def get_shapes(tensors):
    return [t.size() for t in tensors]


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

    def pad_right_params(self, params):
        padded = torch.zeros(457, 192).to("cpu")

        offset = 0
        for n, t in params:
            if len(t.size()) == 2:
                padded[offset:offset+t.size(0), :t.size(1)] += t
                offset += t.size(0)
            elif len(t.size()) == 1:
                padded[offset:offset+1, :t.size(-1)] += t
                offset += 1

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

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, hidden_dim):
        super(RNN, self).__init__()

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 5)

    def forward(self, obs, hidden_state=None):
        x = F.relu(self.fc1(obs), inplace=False)

        if hidden_state:
            h_in = hidden_state.reshape(-1, 64)
            h = self.rnn(x, h_in)
        else:
            hidden_state = torch.zeros(32, 64, device="cpu")
            h = self.rnn(x, hidden_state)

        q = self.fc2(h)
        return q, h

class BatchRNN(nn.Module):
    def __init__(self, params, wrapper):
        super(BatchRNN, self).__init__()
        self.policy

        for p in params:
            pass

    def forward(self, p):
        pass


def policy_output_loss(reconstructeds, originals):
    random_input = torch.randn(32, 43).to("cpu")

    policy_out_loss = 0.0
    for reconstructed, original in zip(reconstructeds, originals):
        with torch.no_grad():
            q_orig, _ = original(random_input)

        q_recon, _ = reconstructed(random_input)
        policy_out_loss += F.mse_loss(q_recon, q_orig)

    return policy_out_loss


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    # device = torch.device("cuda" if True else "cpu")
    device = torch.device("cuda:0")
    device = torch.device("cpu")

    weight_file = '/home/joosep/PycharmProjects/StarCraft/latest/seed-scan/1614471632_5x5_eval_12x12_10A5P_fullmono_notime_noreset_epsilon_eval_seed_109/params/50_rnn_net_params.pkl'
    params = torch.load(weight_file)

    wrapper = PolicyWrapper(params)

    policy_parameters, targets = get_data()
    padded = torch.cat(wrapper.pad_right(params), dim=0)
    policy = wrapper.to_policy(padded)
    print(padded.size())

    for (no, original), (nr, reconstructed) in zip(params.items(), policy.items()):
        assert no == nr
        assert torch.sum(original - reconstructed) == 0, torch.sum(original - reconstructed)

    original_policies = []
    policies = []

    for w in policy_parameters:
        p = RNN(43, 64).to(device)
        p.load_state_dict(w)
        policies.append(p)

        p = RNN(43, 64).to(device)
        p.load_state_dict(w)
        original_policies.append(p)

    original_policies = np.array(original_policies)
    policies = np.array(policies)


    padded_weights = torch.cat(list(map(lambda x: wrapper.pad_right_params(x.named_parameters()).unsqueeze(dim=0), policies)), dim=0).to(device)

    targets = torch.cat([torch.from_numpy(t).unsqueeze(dim=0) for t in targets], dim=0).to(device)

    # z score
    # targets = targets - torch.mean(targets)
    targets = targets / torch.max(targets)

    mask = torch.cat(wrapper.masks, dim=0).to(device)

    if False and os.path.isfile('model.pt'):
        model = GRUVAE().to(device)
        weights = torch.load('model.pt', map_location=device)
        model.load_state_dict(weights)
        noise = torch.randn((1024, padded.size(0), 4)).to(device)
        x = model.decode(noise)
        best = torch.argmax(model.predict_performance(noise))
        test_policy = wrapper.to_policy(x[best].to("cpu"))
        torch.save(test_policy, "weights.pt")
        exit(0)

    model = GRUVAE(device).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4, weight_decay=1e-8)
    # reconstruction warmup https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
    batch_idxes = np.arange(len(policies))
    rg =  default_rng(12345)

    for epoch in range(50):
        rg.shuffle(batch_idxes)
        batch_idxes = np.array_split(np.arange(len(policies)), 32)

        train_loss = 0.0
        mse_loss = 0.0
        perf_loss = 0.0
        policy_loss = 0.0

        for batch_idx in batch_idxes:
            x = padded_weights[batch_idx]
            target = targets[batch_idx]

            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse

            ys = model.predict_performance(zs)
            perf_pred_loss = F.mse_loss(ys, target)
            loss += perf_pred_loss

            policy_out_loss = policy_output_loss(policies[batch_idx], original_policies[batch_idx])
            loss += policy_out_loss

            loss.backward(retain_graph=True)

            train_loss += loss.item()
            mse_loss += mse.item()
            perf_loss += perf_pred_loss.item()
            policy_loss += policy_out_loss.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}, Perf Pred Loss {:.6f}, Policy Out Loss {:.6f}'.format(
                epoch,
                train_loss / len(batch_idxes),
                mse_loss / len(batch_idxes),
                perf_loss / len(batch_idxes),
                policy_loss / len(batch_idxes)
            )
        )

    # all together
    for epoch in range(50):
        train_loss = 0.0
        mse_loss = 0.0
        kld_loss = 0.0
        perf_loss =0.0
        policy_loss = 0.0

        for batch_idx in batch_idxes:
            x = padded_weights[batch_idx]
            target = targets[batch_idx]

            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse + kld

            ys = model.predict_performance(zs)
            perf_pred_loss = F.mse_loss(ys, target)
            loss += perf_pred_loss


            policy_out_loss = policy_output_loss(policies[batch_idx], original_policies[batch_idx])
            loss += policy_out_loss
            loss.backward(retain_graph=True)

            train_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            perf_loss += perf_pred_loss.item()
            policy_loss += policy_out_loss.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}, KLD Loss {:.6f}, Perf Pred Loss {:.6f}'.format(
                epoch,
                train_loss / len(batch_idxes),
                mse_loss / len(batch_idxes),
                kld_loss / len(batch_idxes),
                perf_loss / len(batch_idxes)
            )
        )


    model.noise_fixed = False
    # all together
    for epoch in range(200):
        train_loss = 0.0
        mse_loss = 0.0
        kld_loss = 0.0
        perf_loss =0.0
        policy_loss = 0.0

        for batch_idx in batch_idxes:
            x = padded_weights[batch_idx]
            target = targets[batch_idx]

            optimizer.zero_grad()
            recon_batch, mu, logvar, zs = model(x)
            mse, kld = loss_function(recon_batch, x, mu, logvar, mask=mask)
            loss = mse + kld

            ys = model.predict_performance(zs)
            perf_pred_loss = F.mse_loss(ys, target)
            loss += perf_pred_loss

            policy_out_loss = policy_output_loss(policies[batch_idx], original_policies[batch_idx])
            loss += policy_out_loss
            loss.backward(retain_graph=True)

            train_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            perf_loss += perf_pred_loss.item()
            policy_loss += policy_out_loss.item()

            optimizer.step()

        print(
            'Train Epoch: {}\tTotal Loss: {:.6f}, MSE Loss {:.6f}, KLD Loss {:.6f}, Perf Pred Loss {:.6f}'.format(
                epoch,
                train_loss / len(batch_idxes),
                mse_loss / len(batch_idxes),
                kld_loss / len(batch_idxes),
                perf_loss / len(batch_idxes),
            )
        )
    torch.save(model.state_dict(), 'model.pt')
