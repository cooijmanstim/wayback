import contextlib, gc
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import pt, data, util

def emit(h, size):
  return pt.affine(h, size=size, scope="emit")

def compute_loss(e, y):
  targets = y.max(dim=-1)[1].squeeze(dim=1)
  return func.cross_entropy(e, targets)

def emit_sample(h, size):
  p = func.softmax(emit(h, size=size))
  return pt.sample(p, onehotted=True)

def monitor_parameters(parameters):
  for parameter in parameters:
    print("\t%20s %10.8f grad %10.8f"
          % (parameter.name,
             pt.to_numpy(parameter.abs().log1p().mean()),
             pt.to_numpy(parameter.grad.abs().log1p().mean())))

OVERLAP = 1

class Model(util.Factory):
  def train(self, batch, tbptt=32, parameters=None, optimizer=None, after_step_hook=lambda: None):
    self.initialize(len(batch))

    for i, segment in enumerate(pt.segments(batch, length=tbptt, overlap=OVERLAP)):
      with util.timing("step"):
        segment = pt.onehot(segment, size=self.hp.data_dim)
        B, T, D = segment.size()
  
        losses = []
        optimizer.zero_grad()
        for t in range(T - OVERLAP):
          self.transition(segment[:, t])
          e = emit(self.output, size=D)
          losses.append(compute_loss(e, segment[:, t+1]))
  
        losses = torch.stack(losses)
        if np.isnan(pt.to_numpy(losses)).any():
          raise ValueError("NaN encountered")
        loss = losses.mean()
        print("i %i loss %5.3f" % (i, pt.to_numpy(loss)))
  
        loss.backward(retain_variables=True)
        torch.nn.utils.clip_grad_norm(parameters, 10)
        optimizer.step()
  
        monitor_parameters(parameters)
        self.detach()

      after_step_hook()

  def sample(self, batch, length=100, segment_length=None):
    self.initialize(len(batch), inference=True)

    # conditioning on given batch of examples
    if segment_length is None:
      segment_length = max(map(len, batch))
    for i, segment in enumerate(pt.segments(batch, length=segment_length)):
      segment = pt.onehot(segment, size=self.hp.data_dim)
      B, T, D = segment.size()
      for t in range(T):
        self.transition(segment[:, t])

    # running free
    ys = []
    for t in range(length):
      ys.append(emit_sample(self.output, D))
      self.transition(ys[-1])
    ys = torch.stack(ys, dim=1)

    return [data.Example([pt.to_numpy(y)]) for y in pt.unhot(ys, dim=-1)]

  def initialize(self, batch_size, inference=False): raise NotImplementedError()
  def transition(self, *xs):                         raise NotImplementedError()
  def detach(self):                                  raise NotImplementedError()
  @property
  def output(self):                                  raise NotImplementedError()

class LSTM(Model):
  key = "lstm"

  def __init__(self, hp):
    self.hp = hp
    self.activation = pt.get_activation(self.hp.activation)
    self.affinity = pt.get_affinity(self.hp.affinity)

  def initialize(self, batch_size, inference=False):
    self.h, self.c = [Variable(torch.zeros(batch_size, self.hp.size).cuda(),
                               volatile=inference)
                      for _ in range(2)]

  @pt.namespaced
  def transition(self, *xs):
    fuu = self.affinity(self.h, *xs, size=4 * self.hp.size, scope="gates")
    i, f, g, o = [fuu[:, i * self.hp.size:(i + 1) * self.hp.size] for i in range(4)]
    self.c = f.sigmoid() * self.c + i.sigmoid() * self.activation(g)
    self.h = o.sigmoid() * self.activation(self.c)
    return self.output

  def detach(self):
    self.h = Variable(self.h.data)
    self.c = Variable(self.c.data)

  @property
  def output(self):
    return self.h

class RNN(Model):
  key = "rnn"

  def __init__(self, hp):
    self.hp = hp
    self.activation = pt.get_activation(self.hp.activation)
    self.affinity = pt.get_affinity(self.hp.affinity)

  def initialize(self, batch_size, inference=False):
    self.h = Variable(torch.zeros(batch_size, self.size).cuda(), volatile=inference)

  @pt.namespaced
  def transition(self, *xs):
    self.h = self.activation(self.affinity(*xs, self.h, size=self.size, normalized=self.normalized, scope="rnn"))
    return self.output

  def detach(self):
    self.h = Variable(self.h.data)

  @property
  def output(self):
    return self.h

class Wayback(Model):
  key = "wayback"

  def __init__(self, hp):
    self.hp = hp

    assert len(hp.layers) == len(hp.strides)

    self.layers = [Model.make(layerhp.kind, layerhp) for layerhp in hp.layers]
    self.strides = np.asarray(hp.strides)

    # cutoffs as specified in hp is measured in layer strides and is relative to the end.
    # self.cutoffs is measured in time steps from the beggining.
    things = hp.cutoffs * self.strides
    self.period = things[-1]
    self.cutoffs = self.period - things
    self.optimal_cutoff = max(things)

  def initialize(self, batch_size, inference=False):
    self.time = 0
    for layer in self.layers:
      layer.initialize(batch_size, inference=inference)

  @pt.namespaced
  def transition(self, *xs):
    for layer, cutoff in zip(self.layers, self.cutoffs):
      if self.time < cutoff:
        layer.detach()
    for i in reversed(range(len(self.layers))):
      if self.time % self.strides[i] == 0:
        # TODO explore use of attention going upward
        context = self.layers if self.hp.vskip else self.layers[i-1:i+1]
        self.layers[i].transition(*(list(xs) + [layer.output for layer in context]), scope=str(i))
    self.time = (self.time + 1) % self.period
    return self.output

  def detach(self):
    for layer in self.layers:
      layer.detach()

  @property
  def output(self):
    # ensure gradient cutoff is not bypassed by the client
    output = torch.cat([layer.output for layer in self.layers], dim=-1)
    if self.time < self.cutoffs[0]:
      output.detach()
    return output

  def train(self, batch, *args, **kwargs):
    # plus one because we need a target for the last timestep
    min_length = self.optimal_cutoff + OVERLAP
    shortest = min(len(example) for example in batch)
    if shortest < min_length:
      raise ValueError("example of length %i is too short for wayback; need at least %i"
                       % (shortest, min_length))
    if kwargs.get("tbptt", 0) != self.optimal_cutoff:
      print("correcting tbptt length to optimal %i" % self.optimal_cutoff)
      kwargs["tbptt"] = self.optimal_cutoff
    return super().train(batch, *args, **kwargs)

make = Model.make
