import contextlib
import functools as ft, itertools as it
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn.init as initializers
import torch.nn.functional as func
from torch.autograd import Variable

class Parameters(object):
  def __init__(self):
    self.parameters = ordict()
    self.prefixes = []
    self._frozen = False

  def get(self, name, shape, init, allow_reuse=False):
    name = ".".join(self.prefixes + [name])
    if name in self.parameters:
      if shape != list(self.parameters[name].size()):
        raise ValueError("attempt to recreate parameter with different shape (old: %s new: %s)"
                         % (self.parameters[name].size(), shape))
    else:
      if self._frozen:
        raise RuntimeError("attempt to instantiate parameter '%s' while frozen" % name)
      ugh = torch.cuda.FloatTensor(*shape)
      if not callable(init):
        init = ft.partial(initializers.constant, val=init)
      init(ugh)
      self.create(name, ugh)
    assert not np.isnan(to_numpy(self.parameters[name])).any()
    return self.parameters[name]

  def load_from(self, path):
    data = np.load(path)
    # sort because apparently order is nondeterministic through np.savez/np.load
    for name, value in sorted(data.items(), key=lambda item: item[0]):
      self.create(name, from_numpy(value))

  def save_to(self, path):
    data = ordict((name, to_numpy(parameter))
                  for name, parameter in self.parameters.items())
    np.savez_compressed(path, **data)

  def create(self, name, tensor):
    assert not self._frozen
    assert name not in self.parameters
    parameter = Variable(tensor, requires_grad=True)
    parameter.name = name
    self.parameters[name] = parameter
    print(name, "x".join(map(str, tensor.size())))
    return parameter

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
        with self.namespace(kwargs.pop("scope")):
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

def namespaced(fn):
  def wrapped_fn(*args, **kwargs):
    return P.namespaced(fn)(*args, **kwargs)
  return wrapped_fn

@namespaced
def standardize(x, epsilon=1, dim=-1):
  mean = x.mean(dim=dim)
  var = x.var(dim=dim)
  return (x - mean.expand_as(x)) / (var.expand_as(x) + epsilon).sqrt()

@namespaced
def normalize(x, epsilon=1, dim=-1):
  return scale(standardize(x), init=0.1)

@namespaced
def scale(x, dim=-1, init=1):
  g = P.get("g", [x.size()[dim]], init)
  return g.expand_as(x) * x

@namespaced
def bias(x, dim=-1, init=0):
  b = P.get("b", [x.size()[dim]], init)
  return b.expand_as(x) + x

@namespaced
def linear(*xs, dim=None):
  assert dim is not None
  x = torch.cat(xs, dim=-1)
  w = P.get("w", [x.size()[-1], dim], initializers.orthogonal)
  return func.linear(x, w.t())

@namespaced
def weightnormed(*xs, dim=None, epsilon=1e-6):
  assert dim is not None
  x = torch.cat(xs, dim=-1)
  w = P.get("w", [x.size()[-1], dim], initializers.orthogonal)
  g = P.get("g", [dim], ft.partial(initializers.constant, val=0.1))
  w = w * (g / (w.norm(dim=0) + epsilon)).unsqueeze(0).expand_as(w)
  return func.linear(x, w.t())

@namespaced
def affine(*xs, dim=None, normalized=False):
  assert dim is not None
  if normalized:
    return bias(sum(weightnormed(x, dim=dim, scope=str(i)) for i, x in enumerate(xs)))
    #return bias(sum(normalize(linear(x, dim=dim, scope=str(i)), scope=str(i))
    #                for i, x in enumerate(xs)))
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
  u = np.random.random([p.size()[d] if d != dim else 1 for d in range(p.ndimension())]).astype(np.float32)
  split = from_numpy(u).expand_as(cmf) * totalmasses.expand_as(cmf) < cmf
  _, i = split.max(dim=dim)
  i = i.squeeze(dim) # -_-

  return onehot(i, dim=dim, depth=p.size()[dim]) if onehot else i

def onehot(i, depth, dim=None):
  if dim is None:
    dim = i.ndimension()
  size = list(i.size())
  size.insert(dim, depth)
  i = i.unsqueeze(dim)
  return torch.zeros(size).cuda().scatter_(dim, i, 1)

def unhot(p, dim=None):
  if dim is None:
    dim = p.ndimension() - 1
  _, i = p.max(dim=dim)
  i = i.squeeze(dim=dim)
  return i

def selu(x, alpha=1.6732632423543772848170429916717, beta=1.0507009873554804934193349852946):
  return beta * (func.relu(x) + alpha * func.elu(-func.relu(-x)))

def logselu(x):
  return func.relu(x).log1p() + func.elu(-func.relu(-x))

def from_numpy(x):
  return torch.from_numpy(x).cuda()

def to_numpy(x):
  return x.cpu().data.numpy()

def segments(*args, **kwargs):
  for segment in util.segments(*args, **kwargs):
    segment, = util.examples_as_arrays(segment)
    segment = Variable(pt.from_numpy(segment), requires_grad=False)
    yield segment
