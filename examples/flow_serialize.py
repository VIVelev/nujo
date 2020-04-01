from nujo.utils import save_flow, load_flow
import nujo.nn as nn

net = nn.Linear(3, 6) >> nn.Linear(6, 2) >> nn.Linear(2, 1)
save_flow(net)
new_flow = load_flow(net.name + "_parameters.npy")
print(new_flow.parameters)
print(net.parameters)
assert (net.parameters[0][0] == new_flow.parameters[0][0]).all()
