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
  #kwargs.setdefault("transform", transform)
  return patches.Circle(x, **kwargs)

def mkedge(a, b, **kwargs):
  kwargs.setdefault("width", 0.00625)
  kwargs.setdefault("length_includes_head", True)
  #kwargs.setdefault("transform", transform)
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
    self.constant = False

  def connect_from(self, parent):
    self.parents.add(parent)

  @property
  def patches(self):
    return ([mknode(self.x)] +
            [mkedge(a.x, self.x) for a in self.parents])

  @property
  def ancestors(self):
    if self._ancestors is None:
      self._ancestors = (set(self.parents) |
                         set(a for p in self.parents for a in p.ancestors))
    return set(self._ancestors)

  @property
  def subtree(self):
    return set([self]) | set(self.ancestors)

def drawgraph(nodes, ax):
  for node in nodes:
    for patch in node.patches:
      ax.add_artist(patch)

def nodeat(x, nodes):
  return next(node for node in nodes if node.x == x)

# construct complete graph
T = int(np.prod(periods))
states = [Node((-1, y)) for y in range(len(strides))]
nodes = set()
for x in range(T):
  for y, stride in reversed(list(enumerate(strides))):
    if x % stride == 0:
      if y > 0:
        states[y - 1].constant = True
      node = Node((x, y))
      for dy in [-1, 0, 1]:
        if 0 <= y + dy and y + dy < len(states):
          parent = states[y + dy]
          node.connect_from(states[y + dy])
      states[y] = node
      nodes.add(node)

# care only about rightmost bottom node and its ancestors
end = states[0]

forwardnodes = end.subtree

# compute backprop graph
def construct_backward(node, child=None):
  bnode = Node(node.x)
  if child is not None:
    bnode.connect_from(child)
  nodes = set([bnode])
  if not node.constant:
    for parent in node.parents:
      nodes |= construct_backward(parent, node)
  return nodes

backwardnodes = construct_backward(end)

for graph in [nodes, forwardnodes, backwardnodes]:
  fix, ax = plt.subplots(1)
  drawgraph(graph, ax)
  ax.set_aspect('equal', 'datalim')
plt.show()
