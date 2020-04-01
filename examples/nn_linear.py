import nujo as nj
import nujo.nn as nn
import nujo.objective as obj
import nujo.optim as optim

# Define the net and optimizer
net = nn.Linear(3, 6) >> nn.Linear(6, 2) >> nn.Linear(2, 1)
print('Defined net:', net)

loss_fn = obj.L2Loss()
print('Loss:', loss_fn)

optimizer = optim.Adam(net.parameters, lr=0.1)
print('Optimizer:', optimizer)


# Training loop
def train(net, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = net(x)
        # Compute Loss
        loss = loss_fn(output, y)

        # Print the loss every 10th epoch for monitoring
        if epoch % 10 == 0:
            print('EPOCH:', epoch, '| LOSS: ', loss.value)

        # Backprop
        loss.backward()

        # Update
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()


if __name__ == '__main__':
    # Create example data
    x = nj.rand(30, 3, diff=False)
    y = x @ [[2], [3], [4]] - 10

    # Train
    train(net, x, y, 100)
