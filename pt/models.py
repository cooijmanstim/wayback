import contextlib, gc
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

def monitor_parameters(parameters):
  for parameter in parameters:
    print("\t", parameter.name, parameter.abs().log1p().mean().data.numpy())
  nans = set(parameter for parameter in parameters
             if np.isnan(parameter.data.numpy()).any())
  print("nans:", *[parameter.name for parameter in nans])

def monitor_gradients(parameters):
  for parameter in parameters:
    print("\tgrad", parameter.name, parameter.grad.abs().log1p().mean().data.numpy())

class Model(object):
  def train(self, xs, tbptt=32, parameters=None):
    optimizer = torch.optim.Adam(list(parameters), lr=1e-3)
    B, T, D = xs.size()
    self.initialize(B)

    losses = []
    optimizer.zero_grad()
    for i in range(T - 1):
      if i > 0 and i % tbptt == 0:
        loss = torch.stack(losses).mean()
        print("i %i loss %5.3f" % (i, loss.data.numpy()))
        loss.backward(retain_variables=True)
        monitor_gradients(parameters)
        torch.nn.utils.clip_grad_norm(parameters, 10)
        optimizer.step()
        monitor_parameters(parameters)
        losses = []
        optimizer.zero_grad()
        self.detach()
        gc.collect()
      h = self.transition(xs[:, i])
      e = emit(h, dim=D)
      losses.append(compute_loss(e, xs[:, i+1]))
      if np.isnan(losses[-1].data.numpy()):
        raise ValueError("NaN encountered")

  def sample(self, xs, length=100):
    B, T, D = xs.size()
    self.initialize(B, inference=True)
    # conditioning
    for i in range(T):
      h = self.transition(xs[:, i])
    # running free
    ys = []
    for i in range(length):
      ys.append(emit_sample(h, D))
      h = self.transition(Variable(ys[-1], volatile=True))
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

  @pt.namespaced
  def transition(self, *xs):
    fuu = pt.affine(self.h, *xs, dim=4 * self.size, normalized=self.normalized, scope="gates")
    i, f, g, o = [fuu[:, i * self.size:(i + 1) * self.size] for i in range(4)]
    self.c = f.sigmoid() * self.c + i.sigmoid() * g.tanh()
    cout = pt.scale(pt.standardize(self.c), init=0.1) if self.normalized else self.c
    self.h = o.sigmoid() * cout.tanh()
    self.c = cout # normalize cell recurrence
    return self.output

  def detach(self):
    self.h.detach()
    self.c.detach()

  @property
  def output(self):
    return self.h

class RNN(Model):
  def __init__(self, size, normalized):
    self.size = size
    self.normalized = normalized

  def initialize(self, batch_size, inference=False):
    self.h = Variable(torch.zeros(batch_size, self.size), volatile=inference)

  @pt.namespaced
  def transition(self, *xs):
    self.h = pt.affine(*xs, self.h, dim=self.size, normalized=self.normalized, scope="rnn").tanh()
    return self.output

  def detach(self):
    self.h.detach()

  @property
  def output(self):
    return self.h

class Wayback(Model):
  def __init__(self, layers, strides, backprop_lengths, vskip=True):
    self.layers = list(layers)
    self.strides = np.asarray(strides)
    self.backprop_lengths = np.asarray(backprop_lengths)
    self.vskip = vskip

    assert len(self.layers) == len(self.strides)
    assert len(self.layers) == len(self.backprop_lengths)

    things = self.backprop_lengths * self.strides
    self.period = things[-1]
    self.boundaries = self.period - things
    self.optimal_tbptt_length = max(things)

  def initialize(self, batch_size, inference=False):
    self.time = 0
    for layer in self.layers:
      layer.initialize(batch_size, inference=inference)

  @pt.namespaced
  def transition(self, *xs):
    for layer, boundary in zip(self.layers, self.boundaries):
      if self.time < boundary:
        layer.detach()
    for i in reversed(range(len(self.layers))):
      if self.time % self.strides[i] == 0:
        # TODO explore use of attention going upward
        context = self.layers if self.vskip else self.layers[i-1:i+1]
        self.layers[i].transition(*(list(xs) + [layer.output for layer in context]), scope=str(i))
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
    # plus one because we need a target for the last timestep
    min_length = self.optimal_tbptt_length + 1
    if xs.size()[1] < min_length:
      raise ValueError("sequence of length %i is too short for wayback; need at least %i"
                       % (xs.size()[1], min_length))
    if kwargs.get("tbptt", 0) != self.optimal_tbptt_length:
      print("correcting tbptt length to optimal %i" % self.optimal_tbptt_length)
      kwargs["tbptt"] = self.optimal_tbptt_length
    return super().train(xs, *args, **kwargs)
