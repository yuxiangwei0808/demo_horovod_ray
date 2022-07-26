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
from torchvision.models import resnet18, ResNet18_Weights

with open('dict_training.txt', 'r') as f:
    dic = f.read()

rank = []
ip = []
dic = dic.splitlines()
for item in dic:
    if item:
        temp_rank = []
        c_point = item.index(':')
        gpu_rank = item[c_point + 1:]
        gpu_rank = gpu_rank.strip('[')
        gpu_rank = gpu_rank.strip(']')
        gpu_rank_list = gpu_rank.split(',')
        for i in gpu_rank_list:
            temp_rank.append(int(i))

        rank.append(temp_rank)
        ip.append(item[:c_point])


def main():
    hvd.init()

    training_data = datasets.CIFAR10(
        root="~/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.CIFAR10(
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

    print('node_ip, ', ray._private.services.get_node_ip_address())
    current_ip = ray._private.services.get_node_ip_address()
    gpu_id = rank[ip.index(current_ip)]

    local_rank = hvd.local_rank()
    device = gpu_id[local_rank]
    torch.cuda.set_device(device)

    model = resnet18(weights=ResNet18_Weights.DEFAULT).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)


    def on_state_reset():
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * hvd.size()


    @hvd.elastic.run
    def train(state):
        for state.epoch in range(state.epoch, 2 + 1):
            state.model.train()
            train_sampler.set_epoch(state.epoch)
            steps_remaining = len(train_loader) - state.batch
            for state.batch, (data, target) in enumerate(train_loader):
                if state.batch >= steps_remaining:
                    break
                data, target = data.cuda(), target.cuda()
                state.optimizer.zero_grad()
                output = state.model(data)
                loss = criterion(output, target)
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