import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def norm(x):
  return np.sqrt((x ** 2).sum())

def direction(x):
  return np.where(x == 0, 0, x / norm(x))

def arrowbase(a, b, radius):
  return a + radius * direction(b - a)

radius = 0.25

def mknode(x, **kwargs):
  kwargs.setdefault("radius", radius)
  return patches.Circle(x, **kwargs)

def mkedge(a, b, **kwargs):
  kwargs.setdefault("width", 0.00625)
  kwargs.setdefault("length_includes_head", True)
  a, b = np.asarray(a), np.asarray(b)
  dx = b - a
  ray = radius * direction(dx)
  a = a + ray
  dx = dx - 2 * ray
  return patches.FancyArrow(a[0], a[1], dx[0], dx[1], **kwargs)

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

def waybackprop_forward(states, periods):
  strides = np.concatenate([[1], np.cumprod(periods)[:-1]], axis=0)
  T = int(np.prod(periods))
  for x in range(T):
    for y, stride in reversed(list(enumerate(strides))):
      if x % stride != 0:
        # don't update this layer at this time
        continue
      if y > 0:
        # disconnect gradient on layer below
        states[y - 1].constant = True
      node = Node((x, y))
      for dy in [-1, 0, 1]:
        if 0 <= y + dy and y + dy < len(states):
          node.connect_from(states[y + dy])
      states[y] = node
  return states

# construct backprop graph
def backward(node, child=None):
  bnode = Node(node.x)
  if child is not None:
    bnode.connect_from(child)
  nodes = set([bnode])
  if not node.constant:
    for parent in node.parents:
      nodes |= backward(parent, node)
  return nodes

periods = np.array([3] * 3)
periods = np.array([9])
states = [Node((-1, y)) for y in range(len(periods))]
states = waybackprop_forward(states, periods)
# care only about last bottom node and its ancestors
loss = states[0]

forwardnodes = loss.subtree
backwardnodes = backward(loss)

for nodes in [forwardnodes, backwardnodes]:
  fix, ax = plt.subplots(1)
  for node in nodes:
    for patch in node.patches:
      ax.add_patch(patch)
  ax.set_aspect('equal', 'datalim')
  ax.autoscale(True)
plt.tight_layout()
plt.show()
