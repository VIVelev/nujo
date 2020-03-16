import nujo as nj
import nujo.visualization as viz

x = nj.Tensor(10, name='<Tensor> X')
y = 7 * (x**2) + 5 * x + 3

cg_plot = viz.ComputationGraphPlotter().create(y)
cg_plot.view()
