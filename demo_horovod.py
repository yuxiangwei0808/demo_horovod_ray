import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
import os
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
import ray

with open('dict.txt', 'r') as f:
    dic = f.read()

rank = []
ip = []
dic = dic.splitlines()
for item in dic:
    if item:
        c_point = item.index(':')
        rank.append(item[c_point + 1:])
        ip.append(item[:c_point])


def main():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
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

    hvd.init()

    training_data = datasets.FashionMNIST(
        root="~/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="~/data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_data, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=64, sampler=train_sampler)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                              sampler=test_sampler)

    print('gpu_id, ', ray.get_gpu_ids())
    print('node_ip, ', ray._private.services.get_node_ip_address())
    current_ip = ray._private.services.get_node_ip_address()
    gpu_id = rank[ip.index(current_ip)]

    local_rank = hvd.local_rank()
    device = gpu_id[local_rank]

    torch.cuda.set_device(device)
    model = Net().cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)


    def on_state_reset():
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * hvd.size()


    @hvd.elastic.run
    def train(state):
        for state.epoch in range(state.epoch, 100 + 1):
            state.model.train()
            train_sampler.set_epoch(state.epoch)
            steps_remaining = len(train_loader) - state.batch
            for state.batch, (data, target) in enumerate(train_loader):
                if state.batch >= steps_remaining:
                    break
                data, target = data.cuda(), target.cuda()
                state.optimizer.zero_grad()
                output = state.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                state.optimizer.step()
                if state.batch % 10 == 0:

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        state.epoch, state.batch * len(data), len(train_sampler),
                        100.0 * state.batch / len(train_loader), loss.item()))
                state.commit()
            state.batch = 0

    print('start')
    state = hvd.elastic.TorchState(model, optimizer, epoch=1, batch=0)
    state.register_reset_callbacks([on_state_reset])
    train(state)

main()