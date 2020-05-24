import nujo as nj
import nujo.nn as nn

x = nj.rand(1, 1, 5, 5)
conv = nn.Conv2d(1, 3, 2)

x = conv(x)
print(x.shape)
