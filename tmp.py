import nujo as nj
import nujo.nn as nn

x = nj.rand(32, 3, 256, 256)
conv = nn.Conv2d(3, 9, 4, stride=2, padding=1)

x = conv(x)
print(x.shape)
