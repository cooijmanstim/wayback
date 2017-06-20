import contextlib
import functools as ft, itertools as it
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as func
from torch.autograd import Variable

class Parameters(object):
  def __init__(self):
    self.parameters = ordict()
    self.prefixes = []
    self._frozen = False

  def get(self, name, shape, init_fn):
    name = ".".join(self.prefixes + [name])
    if name in self.parameters:
      if shape != self.parameters[name].size():
        raise ValueError("attempt to recreate parameter with different shape (old: %s new: %s)"
                         % (self.parameters[name].size(), shape))
    else:
      if self._frozen:
        raise RuntimeError("attempt to instantiate parameter '%s' while frozen" % name)
      ugh = torch.FloatTensor(*shape)
      init_fn(ugh)
      parameter = Variable(ugh, requires_grad=True)
      parameter.name = name
      self.parameters[name] = parameter
      print(name)
    return self.parameters[name]

  def __iter__(self):
    return iter(self.parameters.values())

  @contextlib.contextmanager
  def namespace(self, prefix):
    self.prefixes.append(prefix)
    yield
    assert self.prefixes.pop() == prefix

  def zero_grad(self):
    for parameter in self.parameters.values():
      if parameter.grad:
        parameter.grad.data.zero_()

  @contextlib.contextmanager
  def frozen(self):
    # to catch accidental creation of new parameters
    _frozen = self._frozen
    self._frozen = True
    yield
    self._frozen = _frozen

  def freeze(self):
    self._frozen = True

  def namespaced(self, fn):
    def namespaced_fn(*args, **kwargs):
      if "scope" in kwargs:
        with PP.namespace(kwargs.pop("scope")):
          return fn(*args, **kwargs)
      else:
        return fn(*args, **kwargs)
    return namespaced_fn

P = None

@contextlib.contextmanager
def parameters():
  global P
  oldP, P = P, Parameters()
  yield P
  P, oldP = oldP, None

def standardize(x, epsilon=1e-6):
  x -= x.mean(dim=-1).expand_as(x)
  x /= x.norm(dim=-1).expand_as(x) + epsilon
  return x

@P.namespaced
def scale(x, dim=-1, init=1):
  g = P.get("g", [x.size()[dim]], ft.partial(init.constant, val=init))
  return g.expand_as(x) * x

@P.namespaced
def bias(x, dim=-1, init=0):
  b = P.get("b", [x.size()[dim]], ft.partial(init.constant, val=init))
  return b.expand_as(x) + x

@P.namespaced
def linear(*xs, dim=None):
  assert dim is not None
  x = torch.cat(xs, dim=-1)
  w = P.get("w", [x.size()[-1], dim], init.orthogonal)
  return F.linear(x, w.t())

@P.namespaced
def affine(*xs, dim=None, normalized=False):
  assert dim is not None
  if normalized:
    return bias(sum(scale(standardize(linear(x, dim=dim, scope=str(i))), init=0.1)
                    for i, x in enumerate(xs)))
  else:
    x = torch.cat(xs, dim=-1)
    return bias(linear(x, dim=dim))

def sample(p, dim=None, temperature=1, onehotted=False):
  assert (p >= 0).prod() == 1 # just making sure we don't put log probabilities in here

  if dim is None:
    dim = p.ndimension() - 1

  if temperature != 1:
    # this is slow
    p = p ** (1. / temperature)

  cmf = p.cumsum(dim=dim)
  totalmasses = cmf[tuple(slice(None) if d != dim else slice(-1, None) for d in range(cmf.ndimension()))]
  u = np.random.random([p.size()[d] if d != dim else 1 for d in range(p.ndimension())])
  _, i = (Tensor(u).expand_as(cmf) * totalmasses.expand_as(cmf) < cmf).max(dim=dim)
  i = i.squeeze(dim)

  return onehot(i, dim=dim, depth=p.size()[dim]) if onehot else i

def onehot(i, depth, dim=None):
  if dim is None:
    dim = i.ndimension()
  size = list(i.size())
  size.insert(dim, depth)
  i = i.unsqueeze(dim)
  return torch.zeros(size).scatter_(dim, i, 1)
