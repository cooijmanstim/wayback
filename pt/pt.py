import contextlib
import functools as ft, itertools as it
from collections import OrderedDict as ordict
import numpy as np
import torch
import torch.nn.init as initializers
import torch.nn.functional as func
from torch.autograd import Variable
import util

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
def scale(x, dim=-1, init=1):
  g = P.get("g", [x.size()[dim]], init)
  return g.expand_as(x) * x

@namespaced
def bias(x, dim=-1, init=0):
  b = P.get("b", [x.size()[dim]], init)
  return b.expand_as(x) + x

@namespaced
def linear(*xs, size=None):
  assert size is not None
  x = torch.cat(xs, dim=-1)
  w = P.get("w", [x.size()[-1], size], initializers.orthogonal)
  return func.linear(x, w.t())

def sample(p, dim=-1, temperature=1, onehotted=False):
  if isinstance(p, Variable):
    # pytorch -__________________-
    return Variable(sample(p.data, dim=dim, temperature=temperature, onehotted=onehotted), requires_grad=False)

  assert (p >= 0).prod() == 1 # just making sure we don't put log probabilities in here

  if temperature == 0:
    return argmax(p, dim=dim, onehotted=onehotted)
  if temperature != 1:
    # this is slow
    p = p ** (1. / temperature)

  cmf = p.cumsum(dim=dim)

  # find total masses by slicing down to the last cmf element along dimension dim
  totals_index = [slice(None)] * cmf.ndimension()
  totals_index[dim] = slice(-1, None)
  totals = cmf[tuple(totals_index)]

  # draw a uniform random number for each variable to be sampled; scale by total
  # mass to take care of normalization
  u = (from_numpy(np.random.random(tuple(totals.size())).astype(np.float32))
       * totals)

  # find the first point at which u <= cmf to find the inverse of the cmf
  lt = u.expand_as(cmf) <= cmf
  assert (to_numpy(lt).sum(axis=dim) > 0).all()

  # max is silently broken, topk seems to work :-/
  x = torch.topk(lt.float(), k=1, dim=dim)[1].squeeze(dim)
  if onehotted:
    x = onehot(x, size=p.size()[dim], dim=dim)
    if x.ndimension() > p.ndimension():
      # pytorch's lack of 0d array strikes again :((((((
      x = x.squeeze()
    assert np.allclose(to_numpy(x.sum(dim=dim)), 1)

  return x

def onehot(i, size, dim=-1):
  if isinstance(i, Variable):
    # pytorch -__________________-
    return Variable(onehot(i.data, size, dim=dim), requires_grad=False)

  if dim < 0:
    # interpret negative dim to count from the end of the onehotted array,
    # which has one more axis
    dim += i.ndimension() + 1

  result_size = list(i.size())
  result_size.insert(dim, size)
  i = i.unsqueeze(dim)

  return torch.zeros(result_size).cuda().scatter_(dim, i.long(), 1.)

def unhot(p, dim=-1):
  return argmax(p, dim=dim)

def argmax(x, dim=-1, onehotted=False):
  i = x.max(dim=dim)[1].squeeze(dim=dim)
  if onehotted:
    p = onehot(i, size=x.size()[dim], dim=dim)
    if x.ndimension() == 1:
      # pytorch doesn't have 0d arrays, so the max over a 1d array is a 1d array of size 1
      # -_____________________________________________________________________________-
      # we can't get rid of it until we have a 2d array again, which p is here
      assert p.ndimension() == 2
      p = p.squeeze()
    assert p.size() == x.size()
    return p
  return i

def selu(x, alpha=1.6732632423543772848170429916717, beta=1.0507009873554804934193349852946):
  return beta * (func.relu(x) + alpha * func.elu(-func.relu(-x)))

def logselu(x):
  return func.relu(x).log1p() + func.elu(-func.relu(-x))

def from_numpy(x):
  return torch.from_numpy(x).cuda()

def to_numpy(x):
  if isinstance(x, Variable):
    x = x.data
  return x.cpu().numpy()

def segments(*args, **kwargs):
  for segment in util.segments(*args, **kwargs):
    segment, = util.examples_as_arrays(segment)
    segment = Variable(from_numpy(segment), requires_grad=False)
    yield segment

def get_activation(designator):
  if callable(designator):
    return designator
  return dict(relu=func.relu,
              tanh=func.tanh,
              sigmoid=func.sigmoid,
              elu=func.elu,
              selu=selu,
              logselu=logselu)[designator]

affinities = dict()

def affinity(key):
  def decorator(fn):
    affinities[key] = fn
    return fn
  return decorator

def get_affinity(designator):
  if callable(designator):
    return designator
  return affinities[designator]

@affinity("plain")
@namespaced
def plain_affinity(*xs, size=None):
  assert size is not None
  x = torch.cat(xs, dim=-1)
  return bias(linear(x, size=size))

@affinity("weightnorm")
@namespaced
def weightnorm_affinity(*xs, size=None, epsilon=1e-6):
  assert size is not None
  zs = []
  for i, x in enumerate(xs):
    with P.namespace(str(i)):
      w = P.get("w", [x.size()[-1], size], initializers.orthogonal)
      g = P.get("g", [size], ft.partial(initializers.constant, val=0.1))
      w = w * (g / (w.norm(dim=0) + epsilon)).unsqueeze(0).expand_as(w)
      z = func.linear(x, w.t())
      zs.append(z)
  return bias(sum(zs))

@affinity("layernorm")
@namespaced
def layernorm_affinity(*xs, size=None, epsilon=1e-6):
  assert size is not None
  ys = []
  for i, x in enumerate(xs):
    with P.namespace(str(i)):
      z = standardize(linear(x, size=size), dim=-1, epsilon=epsilon)
      y = scale(z, init=0.1)
      ys.append(y)
  return bias(sum(ys))

@affinity("condlayernorm")
@namespaced
def condlayernorm_affinity(*xs, size=None, epsilon=1e-6):
  assert size is not None
  zs = []
  ys = []
  for i, x in enumerate(xs):
    with P.namespace(str(i)):
      z = standardize(linear(x, size=size), dim=-1, epsilon=epsilon)
      zs.append(z)
  for i, z in enumerate(zs):
    with P.namespace(str(i)):
      scale = plain_affinity(*zs, size=size, scope="scale")
      y = z * scale
      ys.append(y)
  return sum(ys) + plain_affinity(*zs, size=size, scope="bias")

affine = plain_affinity
