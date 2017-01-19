from collections import defaultdict as ddict
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

  if False:
    def __hash__(self):
      return hash(self.x)
  
    def __eq__(self, other):
      if not isinstance(other, Node):
        return False
      return self.x == other.x
  
    def __neq__(self, other):
      return not self.eq(other)

  @property
  def ancestors(self):
    if self._ancestors is None:
      self._ancestors = (set(self.parents) |
                         set(a for p in self.parents for a in p.ancestors))
    return set(self._ancestors)

  @property
  def subtree(self):
    return set([self]) | set(self.ancestors)

  def __repr__(self):
    return repr(self.x)

def waybackprop_forward(states, strides, length):
  states = list(states)
  for x in range(length):
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

# backward creates multiple nodes with the same x but different sets
# of parents/children :-( merge these after the fact i guess
def merge(nodes):
  adjacency = ddict(set)
  for node in nodes:
    adjacency[node.x].update(p.x for p in node.parents)
  newnodes = dict()
  for a in adjacency.keys():
    newnodes[a] = Node(a)
  for a, bs in adjacency.items():
    for b in bs:
      newnodes[a].connect_from(newnodes[b])
  return newnodes.values()

def schedule(nodes):
  # toposort-like sequence of sets of nodes in order of earliest possible computation
  left = ft.reduce(set.union, [node.ancestors for node in nodes], set(nodes))
  done = set()
  schedule = []
  while left:
    raa = set(node for node in left if node.parents <= done)
    schedule.append(raa)
    done |= raa
    left -= raa
  return schedule

strides = np.array([1, 5, 25])
length = 50
initial_states = [Node((-stride, y)) for y, stride in enumerate(strides)]
final_states = waybackprop_forward(initial_states, strides, length)

# compute gradient of last state, irl would be gradient of loss which is an aggregate of statewise predictions
loss = final_states[0]
forwardnodes = loss.subtree
#forwardschedule = schedule([loss])

# actually we don't want to compute according to schedule which computes
# upper nodes as soon as their dependencies are available;
# instead we want to go left-to-right top-to-bottom, time-major order
# as it communicates the idea better.
# basically exactly according to the algorithm that constructs the forward graph,
# but let's hack up a new forwardschedule with that order.
forwardschedule = [set([n]) for n in sorted(sorted(forwardnodes,
                                                   key=lambda n: -n.x[1]),
                                            key=lambda n: n.x[0])]

backwardnodes = merge(backward(loss))
#backwardschedule = schedule(backwardnodes)

# similarly backwardschedule.
backwardschedule = list(reversed([set([n]) for n in sorted(sorted(backwardnodes,
                                                                  key=lambda n: -n.x[1]),
                                                           key=lambda n: n.x[0])]))

# nodes whose values are stored in memory; forward equivalents of backward nodes
memorized_nodes = set(node for node in forwardnodes
                      if any(bnode.x == node.x for bnode in backwardnodes))

# cumulative forward schedule
cfws = []
cfwn = set()
for nodes in forwardschedule:
  cfwn |= nodes
  cfws.append(set(cfwn))
  # accumulate only stored nodes
  #cfwn &= memorized_nodes

# cumulative backward schedule
cbws = []
cbwn = set()
for nodes in backwardschedule:
  cbwn |= nodes
  cbws.append(set(cbwn))

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
  color = Colors.dullblue
  if active:
    color = Colors.magenta if backward else Colors.blue
  return patches.FancyArrow(a[0], a[1], dx[0], dx[1],
                            facecolor=color,
                            edgecolor=color,
                            **kwargs)

def draw_backward_subtree():
  fig, ax = plt.subplots(1)
  if not backwardnodes:
    import pdb; pdb.set_trace()
  # draw backward structure
  for node in backwardnodes:
    ax.add_patch(node_patch(node, active=False))
    for parent in node.parents:
      ax.add_patch(edge_patch(parent, node, active=False))
  ax.set_aspect('equal', 'datalim')
  ax.autoscale(True)
  fig, ax = plt.subplots(1)
  # draw backward structure
  for nodes in backwardschedule:
    for node in nodes:
      ax.add_patch(node_patch(node, active=False))
      for parent in node.parents:
        ax.add_patch(edge_patch(parent, node, active=False))
  ax.set_aspect('equal', 'datalim')
  ax.autoscale(True)

def draw_animation():
  fig, ax = plt.subplots(1)
  # draw inactive backdrop
  for nodes in forwardschedule:
    for node in nodes:
      ax.add_patch(node_patch(node, active=False))
      for parent in node.parents:
        ax.add_patch(edge_patch(parent, node, active=False))
  
  # animate forward
  artistsequence = []
  for nodes in cfws:
    artists = []
    for node in nodes:
      artists.append(node_patch(node, active=True))
      for parent in node.parents:
        artists.append(edge_patch(parent, node, active=True))
    artistsequence.append(artists)
    # associate artists with ax
    for artist in artists:
      ax.add_patch(artist)

  memorized_artists = artistsequence[-1]

  # animate backward
  for nodes in cbws:
    artists = []
    # keep showing stored nodes
    artists.extend(memorized_artists)
    for node in nodes:
      artists.append(node_patch(node, active=True, backward=True))
      for parent in node.parents:
        artists.append(edge_patch(parent, node, active=True, backward=True))
    artistsequence.append(artists)
    # associate artists with ax
    for artist in artists:
      ax.add_patch(artist)

  anim = animation.ArtistAnimation(fig, artistsequence, interval=100, repeat_delay=1000, blit=True)
  ax.set_aspect('equal', 'datalim')
  ax.autoscale(True)

  # setting an outer scope variable from within a closure is STILL broken in python 3!
  paused = [False]
  def handle_key_press(event):
    if event.key == " ":
      if paused[0]:
        anim.event_source.start()
        paused[0] = False
      else:
        anim.event_source.stop()
        paused[0] = True
  fig.canvas.mpl_connect("key_press_event", handle_key_press)

  # must keep reference to the animation object or it will die :/
  return anim

#draw_backward_subtree()
anim = draw_animation()
plt.tight_layout()
plt.show()
