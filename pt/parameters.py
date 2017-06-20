import contextlib
from collections import OrderedDict as ordict
import numpy as np
from torch.autograd import Variable
from torch import Tensor # alright then

class Parameters(object):
  def __init__(self):
    self.parameters = ordict()
    self.prefixes = []
    self._frozen = False

  def get(self, name, init):
    name = ".".join(self.prefixes + [name])
    if name in self.parameters:
      if init.shape != self.parameters[name].size():
        raise ValueError("attempt to recreate parameter with different shape (old: %s new: %s)"
                         % (self.parameters[name].size(), init.shape))
    else:
      if self._frozen:
        raise RuntimeError("attempt to instantiate parameter '%s' while frozen" % name)
      parameter = Variable(Tensor(np.asarray(init).astype(np.float32)), requires_grad=True)
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
