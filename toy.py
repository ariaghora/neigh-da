import torch
from abon_toolkit.kernels import euclidean_distances

a = torch.FloatTensor([[0,0,1], [0,0,1]])
b = a.clone() + torch.FloatTensor([[0, 0, 0], [2,0,1]])
b = b / b.sum(1, keepdim=True)

print(a, b)

kld = torch.nn.KLDivLoss(reduction='batchmean')

# print('L:', ce(a.log_softmax(1), b))
print(euclidean_distances(a, b))