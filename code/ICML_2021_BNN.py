# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import itertools
import copy
import os.path
from os import path
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Save the path to store the data
path_folder = '/workdir/data_dg_lmc_icml/mnist_experiment/dglmc'
#print("File exists:"+str(path.exists(path_folder)))

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# If true, we save the weights and samples
save_weights = True

# Number of update for theta
num_epochs = 12000

# Number of worker to distribute the data
num_workers = 50

# Number of local updates before exchanging with the servor
N = 10 * np.ones(num_workers, dtype=int)

# Define the parameter rho_i in function of the worker
rho = .02 * torch.ones(num_workers).to(device)

# Define the harmonic average of rho_i
rho_hm = 1 / torch.sum(1 / rho).to(device)

# Multiplicative constant for gamma_i
gamma = .25 * rho  # / torch.from_numpy(N).type(torch.FloatTensor).to(device)


seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

"""Dataset"""

# define the classes
classes = list(range(0, 10))  # [c0, c1]

# Load the dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

trainset = torchvision.datasets.MNIST(root='/workdir/mnist', train=True,
                                      download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='/workdir/mnist', train=False,
                                     download=True, transform=transform)

# Normalize the dataset
X_train, Y_train = trainset.data, trainset.targets
X_test, Y_test = testset.data, testset.targets
X_train, X_test = X_train.float() / 255, X_test.float() / 255
X_train, X_test = (X_train - X_train.mean()) / \
    X_train.std(), (X_test - X_train.mean()) / X_train.std()
X_train, X_test = X_train.unsqueeze(dim=1), X_test.unsqueeze(dim=1)


class DataLoader:

    def __init__(self, my_iter, data_size):
        self.it = 0
        self.my_iter = itertools.cycle(my_iter)
        self.data_size = data_size

    def __iter__(self):
        self.it = 0
        return self

    def __next__(self):
        if self.it < self.data_size:
            self.it += 1
            return next(self.my_iter)
        else:
            self.it = 0
            raise StopIteration

    next = __next__

    def __len__(self):
        len_ = 0
        for j in range(self.data_size):
            len_ += len(next(self.my_iter)[1])
        return len_


def torch_split(tensor, num_sections=1):
    remainder = len(tensor) % num_sections
    div_points = len(tensor) // num_sections * np.arange(num_sections + 1)
    if remainder != 0:
        div_points[- remainder:] += np.arange(1, remainder + 1, dtype=np.int16)
    sub_tensor = []
    for i in range(num_sections):
        start = div_points[i]
        end = div_points[i + 1]
        sub_tensor.append(tensor[start: end])
    return sub_tensor


#
iter_train = zip(torch_split(X_train, num_workers),
                 torch_split(Y_train, num_workers))
iter_test = zip(torch_split(X_test, num_workers),
                torch_split(Y_test, num_workers))

# Define the train and test loader
trainloader = DataLoader(iter_train, num_workers)
testloader = DataLoader(iter_test, num_workers)

# define the classes
classes = list(range(0, 10))  # [c0, c1]


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))
    plt.axis(False)


# # Get some random training images
# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # Show images
# imshow(torchvision.utils.make_grid(images[:4]))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


""" Neural Network"""


class BNN(nn.Module):

    def __init__(self):
        super(BNN, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)  # output: log(p(y=i|x))


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        # torch.nn.init.xavier_uniform_(m.weight)
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)


# List of neural networks : each worker have its own weights
net_list = []

for i in range(num_workers + 1):
    # Define the choosen architecture
    net = BNN()
    net.to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    # store the neural network
    net_list.append(net)

# Initialized the weights
net_list[-1].apply(init_weights)

# Load the weights of theta to each worker
for i in range(num_workers):
    net_list[i].load_state_dict(net_list[-1].state_dict())


# To count the total number of parameters
def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


print('Number of parameters for one neural network is: %s.' %
      count_parameters(net))

"""Training stage"""

# Define the loss function
criterion = nn.CrossEntropyLoss(reduction='sum')

# L2 penalization corresponding to a gaussian prior
l2_reg = .01 / num_workers

# Will contain the losses
losses_test = []

# Will contain the accuracies
accuracies_test = []

# Will contain the sampled parameters
theta_net = []

# Define the file title
title = '/seed=' + str(seed) + '_workers=' + str(num_workers) \
        + '-N={0:.1E}_gamma={1:.1E}_rho={2:.1E}'.format(N[0], gamma[0], rho[0])

# Define epoch_save which contains the epochs for which we want to save the parameters
epoch_save = [0, 1]
tau = .515
while epoch_save[-1] < num_epochs:
    new_el = tau * np.sum(epoch_save[-2:])
    epoch_save.append(new_el)
epoch_save = np.unique(np.round(epoch_save[:-1] + [num_epochs]))

# Save statistics before the training stage
correct = 0
total = 0
loss_test = 0
with torch.no_grad():
    # compute the prior
    for param in net.parameters():
        loss_test += num_workers * l2_reg * torch.norm(param) ** 2
    for images, labels in trainloader:
        # to run on gpu
        images, labels = images.to(device), labels.to(device)
        # compute the predictions
        outputs = net_list[-1](images)  # loss = - log(y_i|x_i)
        # predict the most likely label
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_test += criterion(outputs, labels).item()
# Save the loss
loss_test = loss_test.cpu()
losses_test.append(loss_test.numpy())
# Save the accuracy
accuracies_test.append(100 * correct / total)
# Print the loss and accuracy
print(losses_test[-1], accuracies_test[-1])

"""Now, we launch the training stage."""

for epoch in range(num_epochs):

    # theta_param_dict contrain the dictionnary associated to the neural network with parameter theta
    theta_params_dict = dict(net_list[-1].named_parameters())
    for i, (images, labels) in enumerate(trainloader):
        # take the dataset of the i-th worker
        images, labels = images.to(device), labels.to(device)
        # take the neural network of the i-th worker
        net = net_list[i]
        # the i-th worker do N_i local updates
        for k in range(N[i]):
            # zero the parameter gradients
            net.zero_grad()
            # compute the predictions
            outputs = net(images)  # loss = - log(y_i|x_i)
            loss = criterion(outputs, labels)
            # add the gaussian prior
            for param in net.parameters():
                loss += l2_reg * torch.norm(param) ** 2
            # compute the gradient of loss with respect to all Tensors with requires_grad=True
            loss.backward()
            # disable gradient calculation to reduce memory consumption
            with torch.no_grad():
                for name, param in net.named_parameters():
                    # perform the ULA step
                    param.copy_(param + gamma[i] / rho[i] * (theta_params_dict[name] - param)
                                - gamma[i] * param.grad.data + torch.sqrt(2 * gamma[i]) * torch.randn(param.shape).to(
                        device))

    # take the neural network with theta as parameter
    net = net_list[-1]

    # update theta
    with torch.no_grad():
        for name, param_theta in net.named_parameters():
            mu = torch.zeros_like(param_theta)
            for i in range(num_workers):
                param_Z = dict(net_list[i].named_parameters())[name]
                mu += param_Z / rho[i]
            param_theta.copy_(rho_hm * mu + torch.sqrt(rho_hm)
                              * torch.randn(param_theta.shape).to(device))

    # save the parameter theta
    save_theta = (epoch > .1 * num_epochs)
    if save_theta:
        theta_net.append(copy.deepcopy(net))

    # save the parameters only if save_weights = True
    if (epoch % 1000 == 999) and save_weights:
        for (i, net) in enumerate(net_list):
            torch.save(net, path_folder + '/worker_samples' +
                       title + '-worker_num_%s' % (i))
        for (i, net) in enumerate(theta_net):
            torch.save(net, path_folder + '/theta_samples' +
                       title + '_' + str(i + 1))

    # print statistics
    if epoch + 1 in epoch_save:

        correct = 0
        total = 0
        loss_test = 0
        with torch.no_grad():
            # compute the prior
            for param in net.parameters():
                loss_test += num_workers * l2_reg * torch.norm(param) ** 2
            for images, labels in trainloader:
                # to run on gpu
                images, labels = images.to(device), labels.to(device)
                # compute the predictions
                outputs = net(images)  # loss = - log(y_i|x_i)
                # predict the most likely label
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_test += criterion(outputs, labels).item()
        # save the loss
        loss_test = loss_test.cpu()
        losses_test.append(loss_test.numpy())
        # save the accuracy
        accuracies_test.append(100 * correct / total)
        print('Epoch number: %s; loss = %s; accuracy %d %%.' %
              (epoch + 1, loss_test, 100 * correct / total))

"""Save the neural network of each worker as well as the neural networks got from the sampled parameters."""

if save_weights:
    for (i, net) in enumerate(net_list):
        torch.save(net, path_folder + '/worker_samples' +
                   title + '-worker_num_%s' % (i))
    for (i, net) in enumerate(theta_net):
        torch.save(net, path_folder + '/theta_samples' +
                   title + '_' + str(i + 1))

# Save the epoch_save
np.save(path_folder + '/epoch_save' + title[1:], epoch_save)

# Save the loss in function of the epoch
np.save(path_folder + '/losses_test' + title[1:], losses_test)

# Save the accuracy in function of the epoch
np.save(path_folder + '/accuracies_test' + title[1:], accuracies_test)

# Save the results
gamma = gamma.cpu()
rho = rho.cpu()
file = open(path_folder + title + '.txt', 'w')
file.write('number epochs = %s,\naccuracy = %s\n' %
           (num_epochs, accuracies_test[-1]))
file.write('N = %s,\ngamma = %s, \nrho = %s, \nloss= %s' %
           (N[0], gamma[0].numpy(), rho[0].numpy(), losses_test[-1]))
file.close()  # to change file access modes
