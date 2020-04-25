import numpy as np
from mnist import MNIST

import nujo.nn as nn
import nujo.objective as obj
import nujo.optim as optim

net = nn.Linear(28 * 28, 20) >> nn.Linear(20, 10) >> nn.Linear(10, 10)
print(f'Defined net: {net}')

loss_fn = obj.L2Loss()
print(f'Loss: {loss_fn}')

optimizer = optim.Adam(net.parameters, lr=0.1)
print(f'Optimizer: {optimizer}')


def train(net, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = net(x)
        # Compute Loss
        loss = loss_fn(output, y)

        # Print the loss for monitoring
        print('EPOCH:', epoch, '| LOSS: ', loss.value)

        # Backprop
        loss.backward()

        # Update
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()


if __name__ == '__main__'
    mndata = MNIST('datasets/MNIST', gz=False)
    img, labels = mndata.load_training()

    images = []
    for i in range(len(img)):
        elem = np.array(img[i]).reshape((len(img[i]), 1))
        images.append(elem)

    images = np.array(images).squeeze()
    labels = np.expand_dims(np.array(labels), -1)

    train(net, images[:32, :], labels[:32], 10)
