from collections import defaultdict as ddict
import functools as ft, itertools as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import seaborn

def main():
  length = 27
  scenario = "tbptt"
  if scenario == "bptt":
    model = Flat(depth=2, backprop_length=None)
  elif scenario == "tbptt":
    model = Flat(depth=2, backprop_length=9)
  elif scenario == "msbptt":
    model = Wayback(strides=[1, 3, 9], truncate=False)
  elif scenario == "mstbptt":
    model = Wayback(strides=[1, 3, 9], truncate=True)
  initial_states = model.initial_states
  final_states = model(initial_states, length)
  loss = model.loss(final_states)
  forwardnodes = loss.subtree
  ymax = max(node.x[1] for node in forwardnodes)
  backwardnodes = backward(loss, xoffset=np.array([-0.5, ymax + 2]))
  schedule = do_schedule(set(forwardnodes) | set(backwardnodes))
  anim = draw_animation(schedule)
  plt.tight_layout()
  plt.show()

class Node(object):
  def __init__(self, x, backward=False, loss=False):
    self.x = tuple(x)
    self.parents = set()
    self.children = set()
    self._ancestors = None
    self.constant = False
    self.backward = backward
    self.loss = loss

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

  def __repr__(self):
    return repr(self.x)

class Flat(object):
  def __init__(self, depth, backprop_length=None):
    self.depth = depth
    self.backprop_length = backprop_length

  @property
  def initial_states(self):
    return [Node((-1, y)) for y in range(self.depth)]

  def loss(self, states):
    output = states[-1]
    loss = Node(output.x + np.array([1, 0]), loss=True)
    loss.connect_from(output)
    return loss

  def __call__(self, states, length):
    backprop_length = self.backprop_length or length # python sucks
    states = list(states)
    for x in range(length):
      for y, _ in enumerate(states):
        if length - x == backprop_length:
          states[y].constant = True
        node = Node((x, y))
        for dy in [-1, 0]:
          if 0 <= y + dy and y + dy < len(states):
            node.connect_from(states[y + dy])
        states[y] = node
    return states

class Wayback(object):
  def __init__(self, strides, truncate=False):
    self.strides = list(strides)
    self.truncate = truncate

  @property
  def initial_states(self):
    return [Node((-stride, y)) for y, stride in enumerate(self.strides)]

  def loss(self, states):
    output = states[0]
    loss = Node(output.x + np.array([1, 0]), loss=True)
    loss.connect_from(output)
    return loss

  def __call__(self, states, length):
    states = list(states)
    for x in range(length):
      for y, stride in reversed(list(enumerate(self.strides))):
        if x % stride != 0:
          # don't update this layer at this time
          continue
        if self.truncate and y > 0:
          # disconnect gradient on layer below
          states[y - 1].constant = True
        node = Node((x, y))
        for dy in [-1, 0, 1]:
          if 0 <= y + dy and y + dy < len(states):
            node.connect_from(states[y + dy])
        states[y] = node
    return states

# construct backprop graph
def backward(node, xoffset=0):
  bnodes = dict()
  def _backward(node, bparent=None):
    # backward node (computes gradient dL/dnode)
    if node.x not in bnodes:
      bnodes[node.x] = Node(node.x + xoffset, backward=True)
      new = True
    else:
      new = False
    bnode = bnodes[node.x]
    if new:
      bnode.connect_from(node)
    if bparent is not None:
      bnode.connect_from(bparent)
    if new and not node.constant:
      for parent in node.parents:
        _backward(parent, bnode)

  # include a backward node for the loss (i.e. dL/dL, which is 1):
  _backward(node)
  # don't include a backward node for the loss:
  #for parent in node.parents:
  #  _backward(parent, bparent=node)
  return set(bnodes.values())

def do_schedule(nodes):
  unknown = ft.reduce(set.union, [node.ancestors for node in nodes], set(nodes))
  justknown = set()
  known = set()
  forgotten = set()
  schedule = []
  while unknown or justknown or known:
    known |= justknown
    incoming = set(node for node in unknown if node.parents <= known)
    justknown = incoming
    unknown -= incoming
    outgoing = set(node for node in known if not node.children & unknown)
    known -= outgoing
    forgotten |= outgoing
    schedule.append(dict(it.chain(
      ((node,   "unknown") for node in   unknown),
      ((node, "justknown") for node in justknown),
      ((node,     "known") for node in     known),
      ((node, "forgotten") for node in forgotten))))
  return schedule


class Colors(object):
  darkblue = (0., 35/255., 61/255.)
  blue = (51/255., 103/255., 214/255.)
  lightblue = (66/255., 133/255., 244/255.)
  aqua = (111/255., 201/255., 198/255.)
  magenta = (194/255., 61/255., 87/255.)
  lightmagenta = (221/255., 79/255., 112/255.)
  gold = "#ffab40"

# memo patch construction functions to avoid creating many duplicate patches.
def memo(f):
  cache = dict()
  def g(*args, **kwargs):
    key = (args, frozenset(kwargs.items()))
    try:
      return cache[key]
    except KeyError:
      cache[key] = f(*args, **kwargs)
      return cache[key]
  return g

saturations = dict(unknown=0.1, justknown=1., known=1., forgotten=0.5)
radius = 0.25

@memo
def node_patch(node, state, **kwargs):
  kwargs.setdefault("radius", radius)
  if node.loss:
    fc = seaborn.desaturate(Colors.gold, saturations[state])
    ec = seaborn.desaturate(Colors.gold, saturations[state])
  elif node.backward:
    fc = seaborn.desaturate(Colors.lightmagenta, saturations[state])
    ec = seaborn.desaturate(Colors.magenta,      saturations[state])
  else:
    fc = seaborn.desaturate(Colors.lightblue, saturations[state])
    ec = seaborn.desaturate(Colors.blue,      saturations[state])
  kwargs["facecolor"] = fc
  kwargs["edgecolor"] = ec
  kwargs["zorder"] = 2
  if state in "known justknown".split():
    kwargs["linewidth"] = 2
  return patches.Circle(node.x, **kwargs)

@memo
def edge_patch(node_a, node_b, state, backward=False, **kwargs):
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
  color = Colors.magenta if backward else Colors.blue
  color = seaborn.desaturate(color, saturations[state])
  kwargs["facecolor"] = color
  kwargs["edgecolor"] = color
  kwargs["zorder"] = 1
  if state == "justknown":
    kwargs["linewidth"] = 1
  assert not np.allclose(dx, 0)
  return patches.FancyArrow(a[0], a[1], dx[0], dx[1], **kwargs)

def draw_animation(schedule):
  fig, ax = plt.subplots(1)
  
  artistsequence = []
  for states in schedule:
    artists = []
    for node, state in states.items():
      artists.append(node_patch(node, state))
      for parent in node.parents:
        artists.append(edge_patch(parent, node, state, backward=node.backward))
    artists = [a for a in artists if a is not None] # sigh, no no-op patch class
    artistsequence.append(artists)

  # leave initial and final state in place for a few frames
  for _ in range(5):
    artistsequence.insert(0, artistsequence[0])
    artistsequence.append(artistsequence[-1])

  # associate (unique) artists with ax
  for artist in set(a for artists in artistsequence for a in artists):
    ax.add_patch(artist)

  # set blit to False to get proper zorders https://github.com/matplotlib/matplotlib/issues/2959
  anim = animation.ArtistAnimation(fig, artistsequence, interval=500, blit=True)
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

if __name__ == "__main__":
  main()
