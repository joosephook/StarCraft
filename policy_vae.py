import torch
import torch.nn.modules as nn
import torch.nn.functional as F


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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class GRUVAE(torch.nn.Module):
    def __init__(self):
        super(GRUVAE, self).__init__()
        # 12288
        # encode
        input_size = 192
        second_size = 64
        latent_size = 4

        self.gru1 = nn.GRU(input_size=input_size, hidden_size=second_size, batch_first=True)
        self.gru2mu = nn.GRU(input_size=second_size, hidden_size=latent_size, batch_first=True)
        self.gru2logvar = nn.GRU(input_size=second_size, hidden_size=latent_size, batch_first=True)

        # decode
        self.gru3 = nn.GRU(input_size=latent_size, hidden_size=second_size, batch_first=True)
        self.gru4 = nn.GRU(input_size=second_size, hidden_size=input_size, batch_first=True)


    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        xs = x
        xs, _ = self.gru1(x) # hidden defaults to 0 if not specified
        xs = F.relu(xs)
        mu_hs, _ = self.gru2mu(xs)
        mu_hs = torch.tanh(mu_hs)# hidden defaults to 0
        logvar_hs, _ = self.gru2logvar(xs)
        logvar_hs = torch.tanh(logvar_hs)

        # return mu_hs.view(-1), logvar_hs.view(-1)
        return mu_hs, logvar_hs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # hs = z.reshape(-1, 1000)
        hs = z
        hs, out = self.gru3(hs)
        hs = F.relu(hs)
        hs, out = self.gru4(hs)
        hs = torch.tanh(hs)

        return hs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x.view, x.view(-1, 784), reduction='sum')
    BCE = F.mse_loss(recon_x, x) # NOT BINARY CROSSENTROPY!

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

from torch.nn.utils.rnn import pad_sequence

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

if __name__ == '__main__':
    from find_weights import get_weights
    torch.manual_seed(0)
    weight_file = '/home/joosep/PycharmProjects/StarCraft/latest/seed-scan/1614471632_5x5_eval_12x12_10A5P_fullmono_notime_noreset_epsilon_eval_seed_109/params/50_rnn_net_params.pkl'
    params= torch.load(weight_file)

    # weights = [w.view(-1, 1) for name, w in params.items()]
    # padded = pad_sequence(weights)
    # padded_weights = padded.squeeze().T
    # print(padded_weights.size())
    # assert padded_weights[0, :2752].equal(weights[0].view(-1))
    # weights = [w for name, w in params.items()]

    padded_weights = map(pad_right, get_weights())
    data = torch.cat([torch.cat(w, dim=0).unsqueeze(dim=0) for w in padded_weights], dim=0)

    # shape = (8, 192)
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(data, data)
    train_loader  = DataLoader(dataset, batch_size=32)
    device = torch.device("cuda" if True else "cpu")

    model = GRUVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(len(dataset))

    for epoch in range(50):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()


        print('Train Epoch: {} [{}/ ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                     batch_idx * len(data),
                                                                     100. * batch_idx / len(train_loader),
                                                                     train_loss / len(dataset)
                                                                     )
              )