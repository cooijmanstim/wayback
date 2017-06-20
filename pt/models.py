import contextlib
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor # alright then
import pt

def emit(h, dim):
  return pt.affine(h, dim=dim, scope="emit")

def compute_loss(e, y):
  targets = y.max(dim=-1)[1].squeeze(dim=1)
  return F.cross_entropy(e, targets)

def emit_sample(h, dim):
  return pt.sample(F.softmax(emit(h, dim=dim)).data, onehotted=True)

class Model(object):
  def train(self, xs, tbptt=32, pp=()):
    adam = torch.optim.Adam(list(pp))
    B, T, D = xs.size()
    self.initialize(B)

    loss = 0
    pp.zero_grad()
    for i in range(T - 1):
      if (i + 1) % tbptt == 0:
        print("loss %5.3f" % loss.data.numpy())
        loss.backward(retain_variables=True)
        adam.step()
        loss = 0
        pp.zero_grad()
        self.detach()
      h = self.transition(x=xs[:, i])
      e = emit(h, dim=D)
      loss += compute_loss(e, xs[:, i+1])

  def sample(self, xs, length=100):
    B, T, D = xs.size()
    self.initialize(B, inference=True)
    # conditioning
    for i in range(T):
      h = self.transition(x=xs[:, i])
    # running free
    ys = []
    for i in range(length):
      ys.append(emit_sample(h, D))
      h = self.transition(x=Variable(ys[-1], volatile=True))
    return torch.stack(ys)

  def initialize(self, batch_size, inference=False): raise NotImplementedError()
  def transition(self, *xs):                         raise NotImplementedError()
  def detach(self):                                  raise NotImplementedError()
  @property
  def output(self):                                  raise NotImplementedError()

class LSTM(Model):
  def __init__(self, size, normalized):
    self.size = size
    self.normalized = normalized

  def initialize(self, batch_size, inference=False):
    self.h, self.c = [Variable(torch.zeros(batch_size, self.size),
                               volatile=inference)
                      for _ in range(2)]

  def transition(self, *xs):
    fuu = pt.affine(self.h, *xs, dim=4 * self.size, normalized=self.normalized, scope="gates")
    i, f, g, o = [fuu[:, i * self.size:(i + 1) * self.size] for i in range(4)]
    self.c = f.sigmoid() * self.c + i.sigmoid() * g.tanh()
    cout = scale(standardize(self.c), init=0.1) if self.normalized else self.c
    self.h = o.sigmoid() * cout.tanh()
    return self.output

  def detach(self):
    self.h.detach()
    self.c.detach()

  @property
  def output(self):
    return self.h

class RNN(Model):
  def __init__(self, size):
    self.size = size

  def initialize(self, batch_size, inference=False):
    self.h = Variable(torch.zeros(batch_size, self.size), volatile=inference)

  def transition(self, *xs):
    self.h = pt.affine(*xs, self.h, dim=self.size, normalized=True, scope="rnn").tanh()
    return self.output

  def detach(self):
    self.h.detach()

  @property
  def output(self):
    return self.h

class Wayback(Model):
  def __init__(self, layers, strides, backprop_lengths, vskip=True):
    self.layers = list(layers)
    self.strides = list(strides)
    self.backprop_lengths = list(backprop_lengths)
    self.vskip = vskip

    assert len(self.layers) == len(self.strides)
    assert len(self.layers) == len(self.backprop_lengths)

    self.period = self.strides[-1]
    self.boundaries = [self.period - backprop_lengths[i] * self.strides[i]
                       for i in range(len(self.backprop_lengths))]

  def initialize(self, batch_size, inference=False):
    self.time = 0
    for layer in self.layers:
      layer.initialize(batch_size, inference=inference)

  def transition(self, *xs):
    for layer, boundary in zip(self.layers, self.boundaries):
      if self.time < boundary:
        layer.detach()
    for i in reversed(range(len(self.layers))):
      if self.time % self.strides[i] == 0:
        context = self.layers if vskip else self.layers[i-1:i+1]
        self.layers[i].transition(*(xs + [layer.output for layer in context]))
    self.time = (self.time + 1) % self.period
    return self.output

  def detach(self):
    for layer in self.layers:
      layer.detach()

  @property
  def output(self):
    # ensure truncation boundary is not bypassed by the client
    output = torch.cat([layer.output for layer in self.layers], dim=-1)
    if self.time < self.boundaries[0]:
      output.detach()
    return output

  def train(self, xs, *args, **kwargs):
    if xs.size()[1] < self.optimal_tbptt_length:
      raise ValueError("sequence of length %i is too short for wayback; need at least %i"
                       % (xs.size()[1], self.optimal_tbptt_length))
    if kwargs.get("tbptt", 0) != self.optimal_tbptt_length:
      print("correcting tbptt length to optimal %i" % self.optimal_tbptt_length)
      kwargs["tbptt"] = self.optimal_tbptt_length
    return super().train(xs, *args, **kwargs)

  @property
  def optimal_tbptt_length(self):
    return max(self.boundaries)
