import nujo as nj
from nujo.utils.viz import ComputationGraphPlotter

x = nj.Tensor(10, name='X')
y = 7 * (x**2) + 5 * x + 3

cg_plot = ComputationGraphPlotter(filename='graph').create(y)
cg_plot.view()
