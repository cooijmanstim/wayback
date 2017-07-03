import contextlib, gc
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import pt, data

def emit(h, dim):
  return pt.affine(h, dim=dim, scope="emit")

def compute_loss(e, y):
  targets = y.max(dim=-1)[1].squeeze(dim=1)
  return func.cross_entropy(e, targets)

def emit_sample(h, dim):
  e = emit(h, dim=dim)
  i = e.max(dim=-1)[1].squeeze(dim=1)
  return pt.onehot(i.data, e.size()[-1])
  return pt.sample(func.softmax(emit(h, dim=dim)).data, onehotted=True)

def monitor_parameters(parameters):
  for parameter in parameters:
    print("\t%20s %10.8f grad %10.8f"
          % (parameter.name,
             pt.to_numpy(parameter.abs().log1p().mean()),
             pt.to_numpy(parameter.grad.abs().log1p().mean())))

OVERLAP = 1

class Model(object):
  def train(self, batch, tbptt=32, parameters=None, optimizer=None, after_step_hook=lambda: None):
    self.initialize(len(batch))

    for i, segment in enumerate(pt.segments(batch, length=tbptt, overlap=OVERLAP)):
      with util.timing("step"):
        segment = pt.onehot(segment, dim=-1)
        B, T, D = segment.size()
  
        losses = []
        optimizer.zero_grad()
        for t in range(T - OVERLAP):
          self.transition(segment[:, t])
          e = emit(self.output, dim=D)
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
      segment = pt.onehot(segment, dim=-1)
      B, T, D = segment.size()
      for t in range(T):
        self.transition(segment[:, t])

    # running free
    ys = []
    for t in range(length):
      ys.append(emit_sample(self.output, D))
      self.transition(Variable(ys[-1], volatile=True))
    ys = torch.stack(ys, dim=1)

    return [data.Example([y]) for y in pt.unhot(ys, dim=-1)]

  def initialize(self, batch_size, inference=False): raise NotImplementedError()
  def transition(self, *xs):                         raise NotImplementedError()
  def detach(self):                                  raise NotImplementedError()
  @property
  def output(self):                                  raise NotImplementedError()

class LSTM(Model):
  def __init__(self, size, normalized, activation=func.tanh):
    self.size = size
    self.normalized = normalized
    self.activation = activation

  def initialize(self, batch_size, inference=False):
    self.h, self.c = [Variable(torch.zeros(batch_size, self.size).cuda(),
                               volatile=inference)
                      for _ in range(2)]

  @pt.namespaced
  def transition(self, *xs):
    fuu = pt.affine(self.h, *xs, dim=4 * self.size, normalized=self.normalized, scope="gates")
    i, f, g, o = [fuu[:, i * self.size:(i + 1) * self.size] for i in range(4)]
    self.c = f.sigmoid() * self.c + i.sigmoid() * self.activation(g)
    cout = pt.scale(pt.standardize(self.c), init=0.1) if self.normalized else self.c
    self.h = o.sigmoid() * self.activation(cout)
    self.c = cout # normalize cell recurrence
    return self.output

  def detach(self):
    self.h = Variable(self.h.data)
    self.c = Variable(self.c.data)
    # this keeps history around somehow
    #self.h.detach()
    #self.c.detach()

  @property
  def output(self):
    return self.h

class RNN(Model):
  def __init__(self, size, normalized, activation=func.tanh):
    self.size = size
    self.normalized = normalized
    self.activation = activation

  def initialize(self, batch_size, inference=False):
    self.h = Variable(torch.zeros(batch_size, self.size).cuda(), volatile=inference)

  @pt.namespaced
  def transition(self, *xs):
    self.h = self.activation(pt.affine(*xs, self.h, dim=self.size, normalized=self.normalized, scope="rnn"))
    return self.output

  def detach(self):
    self.h = Variable(self.h.data)

  @property
  def output(self):
    return self.h

class Wayback(Model):
  def __init__(self, layers, strides, cutoffs, vskip=True):
    self.layers = list(layers)
    self.strides = np.asarray(strides)
    self.vskip = vskip

    assert len(self.layers) == len(self.strides)

    # cutoffs as provided by caller is measured in layer strides and is relative to the end.
    # self.cutoffs is measured in time steps from the beggining.
    things = cutoffs * self.strides
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
        context = self.layers if self.vskip else self.layers[i-1:i+1]
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

  def train(self, xs, *args, **kwargs):
    # plus one because we need a target for the last timestep
    min_length = self.optimal_cutoff + 1
    if xs.size()[1] < min_length:
      raise ValueError("sequence of length %i is too short for wayback; need at least %i"
                       % (xs.size()[1], min_length))
    if kwargs.get("tbptt", 0) != self.optimal_cutoff:
      print("correcting tbptt length to optimal %i" % self.optimal_cutoff)
      kwargs["tbptt"] = self.optimal_cutoff
    return super().train(xs, *args, **kwargs)
