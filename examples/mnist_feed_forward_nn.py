import numpy as np
from mnist import MNIST

import nujo as nj
import nujo.nn as nn
import nujo.objective as obj
import nujo.optim as optim

# TODO: The  neural network now is nothing more than a big linear function
# Use some activations maybe? But which ones?
net = nn.Linear(28 * 28, 20) >> nn.Sigmoid() >> nn.Linear(
    20, 10) >> nn.Sigmoid() >> nn.Linear(10, 10) >> nn.Softmax()

print(f'Defined net: {net}')

# TODO: Maybe try different loss?
# Please look up the difference between classification loss and regression loss
loss_fn = obj.L2Loss()
print(f'Loss: {loss_fn}')

# TODO: Play around with the hyperparameters of Adam
# Maybe try  different optimizers?
optimizer = optim.Adam(net.parameters, lr=0.0001)
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


if __name__ == '__main__':
    mndata = MNIST('datasets/MNIST', gz=False)
    images, labels = mndata.load_training()

    arr = []
    for i in range(len(images)):
        elem = np.array(images[i]).reshape((len(images[i]), 1))
        arr.append(elem)
    images = np.array(arr).squeeze().T
    # print(images.shape)

    labels = np.array(labels).reshape(1, -1)[0]
    print(labels.shape)
    labels = np.eye(max(labels) + 1)[labels]
    print(labels.shape)

    images = nj.Tensor(images[:, :32], name='X_train')
    labels = nj.Tensor(labels[:32].T, name='y_train')

    # TODO: 32 is the batch size in this case
    # Look up what `batch gradient descent` means
    train(net, images, labels, int(1000))
