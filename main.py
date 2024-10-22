import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from abon_toolkit.preprocessing import minibatchify
from abon_toolkit.kernels import rbf_kernel, euclidean_distances, mmd_rbf

torch.manual_seed(13)

cuda_avail = torch.cuda.is_available()
dev = torch.device('cuda' if cuda_avail else 'cpu')


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


def create_mlp(in_size, out_size, out_act, hidden_sizes=(100,), hidden_act=nn.ReLU()):
    sizes = (in_size, *hidden_sizes, out_size)
    pairs = ([(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
    layers = []
    for i, p in enumerate(pairs):
        layers.append(nn.Linear(*p))
        if i < len(pairs) - 1:
            layers.append(hidden_act)
        else:
            layers.append(out_act)
    return nn.Sequential(*layers)


def take_random(x, amount):
    idx = torch.randperm(x.shape[0])[:amount]
    return x[idx]


if __name__ == '__main__':
    S = pd.read_csv('office31-resnet50_feature/webcam_webcam.csv').values
    T = pd.read_csv('office31-resnet50_feature/webcam_amazon.csv').values

    xs, ys = S[:, :-1], S[:, -1]
    xt, yt = T[:, :-1], T[:, -1]
    xs, ys = torch.from_numpy(xs).float(), torch.from_numpy(ys).long()
    xt, yt = torch.from_numpy(xt).float(), torch.from_numpy(yt).long()

    xs, ys = xs.to(dev), ys.to(dev)
    xt, yt = xt.to(dev), yt.to(dev)

    temp = torch.tensor(1.0).to(dev)
    temp.requires_grad_(False)

    emb_net = create_mlp(xs.shape[1], 512, nn.ELU(), hidden_sizes=(1000,), hidden_act=nn.ELU()).to(dev)
    clf_net = create_mlp(512, 31, nn.Identity(), hidden_sizes=(), hidden_act=nn.ELU()).to(dev)

    opt = torch.optim.Adamax([*emb_net.parameters(), *clf_net.parameters(), temp], lr=10e-6)
    ce = nn.CrossEntropyLoss().to(dev)
    sl1 = nn.SmoothL1Loss().to(dev)
    # kld = torch.nn.HingeEmbeddingLoss(reduction='batchmean').to(dev)
    

    minibatches = minibatchify(xs, ys, batch_size=64)

    mmd_rbf = MMD_loss(kernel_num=10)

    max_epochs = 1
    for ep in range(max_epochs):
        losses = []
        for x, y in minibatches:
            opt.zero_grad()

            emb = emb_net(x)
            pred = clf_net(emb)
            pred = torch.softmax(pred/temp, 1)

            xt_b = take_random(xt, x.shape[0])
            emb_t = emb_net(xt_b)

            pred_t = clf_net(emb_t)
            pred_t = torch.softmax(pred_t/temp, 1)
            ent = -torch.mean(torch.sum(pred_t * pred_t.log_softmax(1), 1))

            nei = sl1(euclidean_distances(xt_b, xt_b), euclidean_distances(emb_t, emb_t))
            mmd = mmd_rbf(emb, emb_t)

            loss = ce(pred, y) + mmd + 0.01*nei + ent
            loss.backward()
            losses.append(loss.detach().item())

            opt.step()

        with torch.no_grad():
            emb = emb_net(xt)
            pred_class = torch.softmax(clf_net(emb)/temp, 1).argmax(1).detach().cpu().numpy()
            acc = np.mean(pred_class == yt.cpu().numpy())
            loss = np.mean(losses)
            print(f'\r{ep}: loss: {loss}, acc: {acc}, temp: {temp}', end='')

    torch.save(emb_net, 'emb_net.pth')
    torch.save(clf_net, 'clf_net.pth')
