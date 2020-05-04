import numpy as np
from mnist import MNIST

import nujo as nj
import nujo.nn as nn
import nujo.objective as obj
import nujo.optim as optim
from nujo.utils.viz import ComputationGraphPlotter

net = nn.Linear(28 * 28, 256) >> nn.Sigmoid() \
      >> nn.Linear(256, 128) >> nn.Sigmoid() \
      >> nn.Linear(128, 10) >> nn.Softmax()

print(f'Defined net: {net}')

loss_fn = obj.CrossEntropy()
print(f'Loss: {loss_fn}')

optimizer = optim.Adam(net.parameters, lr=0.01)
print(f'Optimizer: {optimizer}')


def train(net, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = net(x)

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

    return loss


if __name__ == '__main__':
    mndata = MNIST('datasets/MNIST', gz=False)
    images, labels = mndata.load_training()

    arr = []
    for i in range(32):
        elem = np.array(images[i]).reshape(1, -1)
        arr.append(elem[0])
    images = np.array(arr).T / 255

    labels = np.array(labels).reshape(1, -1)[0]
    labels = np.eye(max(labels) + 1)[labels][:32]

    images = nj.Tensor(images, name='X_train')
    labels = nj.Tensor(labels.T, name='y_train')

    loss = train(net, images, labels, 1000)

    # Visualize the Neural Network as a computation graph
    cg_plot = ComputationGraphPlotter(filename='graph').create(loss)
    cg_plot.view()
