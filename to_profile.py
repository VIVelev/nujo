import nujo as nj
from nujo.utils.viz import ComputationGraphPlotter


# Training loop
def train(param, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = param * x
        # Compute Loss
        loss = (output - y)**2

        # Print the loss every 10th epoch for monitoring
        if epoch % 100 == 0:
            print('EPOCH:', epoch, '| LOSS: ', loss.value)

        # Backprop
        loss.backward()

        # Update
        with nj.no_diff():
            param <<= param - 0.01 * param.grad

        # Zero grad
        param.zero_grad()

    return loss


if __name__ == '__main__':
    # Create example data
    x = nj.rand(1, 1, name='X_train')
    y = nj.Tensor(x * nj.rand(1, 1), name='y_train')
    w = nj.randn(1, 1, diff=True, name='weight')

    # Train
    loss = train(w, x, y, 10000)

    # Visualize the Neural Network as a computation graph
    cg_plot = ComputationGraphPlotter(filename='graph').create(loss)
    cg_plot.view()
