import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

def norm(x):
  return np.sqrt((x ** 2).sum())

def direction(x):
  return np.where(x == 0, 0, x / norm(x))

def arrowbase(a, b, radius):
  return a + radius * direction(b - a)

radius = 0.25

transform = transforms.Affine2D()
transform.scale(40)
transform.translate(50, 50)

def mknode(x, **kwargs):
  kwargs.setdefault("radius", radius)
  kwargs.setdefault("transform", transform)
  return patches.Circle(x, **kwargs)

def mkedge(a, b, **kwargs):
  kwargs.setdefault("width", 0.00625)
  kwargs.setdefault("length_includes_head", True)
  kwargs.setdefault("transform", transform)
  a, b = np.asarray(a), np.asarray(b)
  dx = b - a
  ray = radius * direction(dx)
  a = a + ray
  dx = dx - 2 * ray
  return patches.FancyArrow(a[0], a[1], dx[0], dx[1], **kwargs)

nodes = []
edges = []

periods = np.array([3, 3, 3])
strides = np.concatenate([[1], np.cumprod(periods)[:-1]], axis=0)
lastx = np.array([None] * len(periods))

class Node(object):
  def __init__(self, x):
    self.x = x
    self.parents = set()
    self._ancestors = None

  def connect_from(self, parent):
    self.parents.add(parent)

  @property
  def patches(self):
    return ([mknode(self.x)] +
            [mkedge(a.x, self.x) for a in self.parents])

  @property
  def ancestors(self):
    if self._ancestors is None:
      self._ancestors = set(self.parents) | set(a for p in self.parents for a in p.ancestors)
    return set(self._ancestors)

  @property
  def subtree(self):
    return set([self]) | set(self.ancestors)

def graphfigure(nodes):
  fig = plt.figure()
  rawr = lambda *x: fig.patches.extend(x)
  for node in nodes:
    fig.patches.extend(node.patches)

# construct complete graph
nodes = set()
for x in range(int(np.prod(periods)) + 1):
  for y, stride in enumerate(strides):
    if x % stride == 0:
      node = Node((x, y))
      for dy in [-1, 0, 1]:
        if 0 <= y + dy and y + dy < len(lastx) and lastx[y + dy] is not None:
          node.connect_from(next(node for node in nodes
                                 if node.x == (lastx[y + dy], y + dy)))
      lastx[y] = x
      nodes.add(node)

# care only about rightmost bottom node and its ancestors
end = max((node for node in nodes if node.x[1] == 0),
          key=lambda node: node.x[0])

forwardnodes = end.subtree

graphfigure(forwardnodes)
plt.show()

# compute backprop graph from end

