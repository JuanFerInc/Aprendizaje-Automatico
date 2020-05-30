import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os


# https://www.gutenberg.org/ebooks/996
# https://www.gutenberg.org/ebooks/2000


# open text file and read in data as `text`
with open('../input/divinacommedia/dante.txt', 'r') as f:
    text = f.read()