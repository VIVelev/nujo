import numpy as np
from mnist import MNIST

import nujo as nj
import nujo.nn as nn
import nujo.objective as obj
import nujo.optim as optim

net = nn.Linear(28 * 28, 256) >> nn.Sigmoid() \
      >> nn.Linear(256, 128) >> nn.Sigmoid() \
      >> nn.Linear(128, 10) >> nn.Softmax()

print(f'Defined net: {net}')

loss_fn = obj.L2Loss()
print(f'Loss: {loss_fn}')

optimizer = optim.SGD(net.parameters, lr=0.001)
print(f'Optimizer: {optimizer}')


def train(net, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = net(x)
        # print(output)
        # print(y)
        # break
        # Compute Loss
        loss = loss_fn(output, y)

        # Print the loss for monitoring
        if epoch % 100 == 0:
            print(f'EPOCH:\t{epoch}| LOSS:\t{loss.value[0,0]}')

        # Backprop
        loss.backward()

        # Update
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()


if __name__ == '__main__':
    mndata = MNIST('datasets/MNIST', gz=False)
    images, labels = mndata.load_training()

    arr = []
    for i in range(4):
        elem = np.array(images[i]).reshape(1, -1)
        arr.append(elem[0])
    images = np.array(arr).T

    labels = np.array(labels).reshape(1, -1)[0]
    labels = np.eye(max(labels) + 1)[labels][:4]

    images = nj.Tensor(images, name='X_train')
    labels = nj.Tensor(labels.T, name='y_train')

    train(net, images, labels, int(1000))
