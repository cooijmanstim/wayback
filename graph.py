import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import seaborn

class Node(object):
  def __init__(self, x):
    self.x = x
    self.parents = set()
    self.children = set()
    self._ancestors = None
    self.constant = False

  def connect_from(self, parent):
    self.parents.add(parent)
    parent.children.add(self)

  @property
  def ancestors(self):
    if self._ancestors is None:
      self._ancestors = (set(self.parents) |
                         set(a for p in self.parents for a in p.ancestors))
    return set(self._ancestors)

  @property
  def subtree(self):
    return set([self]) | set(self.ancestors)

  @property
  def forwardsequence(self):
    # toposort-like sequence of sets of nodes in order of earliest possible computation
    left = set([self]) | self.ancestors
    done = set([])
    sequence = []
    while left:
      raa = set(node for node in left if node.parents <= done)
      sequence.append(raa)
      done |= raa
      left -= raa
    return sequence

  def __repr__(self):
    return repr(self.x)

def waybackprop_forward(states, periods):
  states = list(states)
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
      nodes |= backward(parent, bnode)
  return nodes

periods = np.array([3] * 3)
initial_states = [Node((-1, y)) for y in range(len(periods))]
final_states = waybackprop_forward(initial_states, periods)

# compute gradient of last state, irl would be gradient of loss which is an aggregate of statewise predictions
loss = final_states[0]
forwardnodes = loss.subtree
forwardsequence = loss.forwardsequence

# actually we don't want to compute according to forwardsequence which computes
# upper nodes as soon as their dependencies are available;
# instead we want to go left-to-right top-to-bottom, time-major order.
# basically exactly according to the algorithm that constructs the forward graph,
# but lacks hack up a new forwardsequence with that order.
forwardsequence = [[n] for n in sorted(sorted(forwardnodes,
                                              key=lambda n: -n.x[1]),
                                       key=lambda n: n.x[0])]

# with respect to initial hidden state
backwardnodes = backward(loss)
# find the parameter node in the backward graph
parameter = next(node for node in backwardnodes if node.x == initial_states[-1].x)
backwardsequence = parameter.forwardsequence

radius = 0.25

class Colors(object):
  darkblue = (0., 35/255., 61/255.)
  blue = (51/255., 103/255., 214/255.)
  lightblue = (66/255., 133/255., 244/255.)
  aqua = (111/255., 201/255., 198/255.)
  magenta = (194/255., 61/255., 87/255.)
  lightmagenta = (221/255., 79/255., 112/255.)
  dullblue = seaborn.desaturate(blue, 0.)
  dulllightblue = seaborn.desaturate(lightblue, 0.)

def node_patch(node, active=True, backward=False, **kwargs):
  kwargs.setdefault("radius", radius)
  fc = Colors.dulllightblue
  ec = Colors.dullblue
  if active:
    if backward:
      fc = Colors.lightmagenta
      ec = Colors.magenta
    else:
      fc = Colors.lightblue
      ec = Colors.blue
    # NOTE: make unmemorized nodes dull again?
    if node.constant:
      fc = Colors.aqua
  kwargs["facecolor"] = fc
  kwargs["edgecolor"] = ec
  return patches.Circle(node.x,
                        **kwargs)

def edge_patch(node_a, node_b, active=True, backward=False, **kwargs):
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
                            facecolor=Colors.blue if active else Colors.dullblue,
                            edgecolor=Colors.blue if active else Colors.dullblue,
                            **kwargs)

fig, ax = plt.subplots(1)

# draw inactive structure
for nodes in forwardsequence:
  for node in nodes:
    ax.add_patch(node_patch(node, active=False))
    for parent in node.parents:
      ax.add_patch(edge_patch(parent, node, active=False))

# animate forward
artistsequence = []
for nodes in forwardsequence:
  artists = []
  for node in nodes:
    artists.append(node_patch(node, active=True))
    for parent in node.parents:
      artists.append(edge_patch(parent, node, active=True))
  artistsequence.append(artists)
  # associate artists with ax
  for artist in artists:
    ax.add_patch(artist)

# animate backward
for nodes in backwardsequence:
  artists = []
  for node in nodes:
    artists.append(node_patch(node, active=True, backward=True))
    for parent in node.parents:
      artists.append(edge_patch(parent, node, active=True, backward=True))
  artistsequence.append(artists)
  # associate artists with ax
  for artist in artists:
    ax.add_patch(artist)

cumulative_artistsequence = []
cumulative_artists = []
for artists in artistsequence:
  cumulative_artists.extend(artists)
  cumulative_artistsequence.append(list(cumulative_artists))
artistsequence = cumulative_artistsequence

yuck = animation.ArtistAnimation(fig, artistsequence, interval=250, repeat_delay=0, blit=True)
ax.set_aspect('equal', 'datalim')
ax.autoscale(True)

plt.tight_layout()
plt.show()
