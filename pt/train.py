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

ncategories = 128
batch_size = 10

numbers = np.array(list(map(ord, text)))
# strip non-ascii
numbers = numbers[numbers < ncategories]

# don't want to run out of memory
#numbers = text[:1000000]
#numbers = numbers[:100000]

# create a batch with staggered copies of the same sequence
offsets = np.arange(0, len(numbers), len(numbers) // batch_size + 1)
xs = np.stack([np.roll(numbers, offset) for offset in offsets])
xs = np.eye(ncategories)[xs]
print(xs.shape)
pfft = Variable(Tensor(xs))

with pt.parameters() as pp:
  model = models.Wayback([models.LSTM(100, normalized=True) for _ in range(2)],
                         strides=[1, 10], backprop_lengths=[100, 100])
  # evaluate model once to ensure parameters are known
  model.sample(pfft[:, :model.optimal_tbptt_length], length=model.optimal_tbptt_length)
  with pp.frozen():
    try:
      for _ in range(100):
        model.train(pfft, parameters=pp)
    except KeyboardInterrupt:
      pass

  cond = pfft[:, :100]
  pred = model.sample(cond, length=1000)

cond = cond.data # FML
def batch_to_strings(zzz):
  zzz = zzz.max(dim=-1)[1].squeeze(dim=-1)
  return ["".join(chr(z) for z in zz) for zz in zzz]

for condex, predex in zip(batch_to_strings(cond), batch_to_strings(pred)):
  print(repr(condex), " ---> ", repr(predex))

import pdb; pdb.set_trace()
