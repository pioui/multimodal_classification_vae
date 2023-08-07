import numpy as np
from tabulate import tabulate

from houston_config import *
import matplotlib

for cname,label in zip(color, labels):
    print('cname:', matplotlib.colors.cnames[cname], label)
