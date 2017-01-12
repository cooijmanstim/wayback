import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node(object):
  def __init__(self, x):
    self.x = x
    self.parents = set()
    self._ancestors = None
    self.constant = False

  def connect_from(self, parent):
    self.parents.add(parent)

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


radius = 0.25

class Colors(object):
  darkblue = (0., 35/255., 61/255.)
  blue = (51/255., 103/255., 214/255.)
  lightblue = (66/255., 133/255., 244/255.)
  aqua = (111/255., 201/255., 198/255.)

def node_patch(node, **kwargs):
  kwargs.setdefault("radius", radius)
  return patches.Circle(node.x,
                        facecolor=Colors.lightblue,
                        edgecolor=Colors.blue,
                        **kwargs)

def edge_patch(node_a, node_b, **kwargs):
  kwargs.setdefault("width", 0.00625)
  kwargs.setdefault("length_includes_head", True)
  a, b = np.asarray(node_a.x), np.asarray(node_b.x)
  # find unit vector from a to b
  dx = b - a
  u = np.where(dx == 0, 0, dx / np.sqrt((dx ** 2).sum()))
  # projection of b onto node_a's perimeter
  a = a + radius * u
  # shorten edge by 2 * radius to go from perimeter to perimeter
  dx = dx - 2 * radius * u
  return patches.FancyArrow(a[0], a[1], dx[0], dx[1],
                            facecolor=Colors.blue,
                            edgecolor=Colors.blue,
                            **kwargs)

for nodes in [forwardnodes, backwardnodes]:
  fix, ax = plt.subplots(1)
  for node in nodes:
    ax.add_patch(node_patch(node))
    for parent in node.parents:
      ax.add_patch(edge_patch(node, parent))
  ax.set_aspect('equal', 'datalim')
  ax.autoscale(True)
plt.tight_layout()
plt.show()
