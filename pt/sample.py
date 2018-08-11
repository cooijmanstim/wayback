import os, sys
import contextlib
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor # alright then
import pt, models

import os, urllib.request
path = "/tmp/pg36.txt"
if not os.path.exists(path):
  urllib.request.urlretrieve("http://www.gutenberg.org/cache/epub/36/pg36.txt", filename=path)
with open(path) as file:
  text = file.read()
print(type(text), text[:30], list(map(ord, text[:30])))

save_path = "parameters.npz"

ncategories = 128
batch_size = 10

numbers = np.array(list(map(ord, text)))
# strip non-ascii
numbers = numbers[numbers < ncategories]

# create a batch with staggered copies of the same sequence
offsets = np.arange(0, len(numbers), len(numbers) // batch_size + 1)
xs = np.stack([np.roll(numbers, offset) for offset in offsets])
xs = np.eye(ncategories)[xs]
print(xs.shape)
pfft = Variable(Tensor(xs))

with pt.parameters() as pp:
  pp.load_from(save_path)
  with pp.frozen():
    strides = [1, 4, 16]
    cutoffs = [64] * len(strides)
    layers = [models.RNN(100, normalized=True, activation=pt.logselu) for _ in strides]
    model = models.Wayback(layers, strides=strides, cutoffs=cutoffs)
  
    print("sampling")
    cond = pfft[:, :50]
    pred = model.sample(cond, length=50)
    cond = cond.data # FML
    def batch_to_strings(zzz):
      zzz = zzz.max(dim=-1)[1].squeeze(dim=-1)
      return ["".join(chr(z) for z in zz) for zz in zzz]
    for condex, predex in zip(batch_to_strings(cond), batch_to_strings(pred)):
      print(repr(condex), " ---> ", repr(predex))

import pdb; pdb.set_trace()
