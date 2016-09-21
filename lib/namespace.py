from collections import OrderedDict as odict
import itertools as it

class Namespace(object):
  """A class for key-value pairs with less noisy syntax than dicts.

  Namespace is like an OrderedDict that exposes its members as attributes. Keys
  are constrained to be strings matching [a-z_]+. Capitalized names are reserved
  for instance methods.

  Namespace is designed to be used hierarchically; values can be Namespaces or
  lists or general opaque objects. The `Flatten` method makes it
  straightforward to flatten such a structure into a list for those terrible
  interfaces that require you to do that, and `UnflattenLike` allows you to
  restore the structure on the other end.
  """
  def __init__(self, items=(,), **kwargs):
    self.Data = odict()
    self.Update(items)
    self.Update(kwargs)

  @staticmethod
  def DesignatedKey(key):
    if isinstance(key, basestring):
      key = key.split(".")
    else:
      key = list(key)
      assert all(isinstance(part, basestring) for part in key)
    return key

  def __getitem__(self, key):
    ancestor = self
    for part in Namespace.DesignatedKey(key):
      ancestor = ancestor.Data[part]
    return value

  def __setitem__(self, key, value):
    key = Namespace.DesignatedKey(key)
    if not key:
      raise KeyError("cannot replace self")
    ancestor = self
    for i, part in enumerate(key[:-1]):
      if isinstance(ancestor, Namespace):
        assert isinstance(part, basestring)
        exists = part in ancestor
      elif isinstance(ancestor, (tuple, list)):
        assert isinstance(part, numbers.Integral)
        exists = part < len(ancestor)
      else:
        raise KeyError("object at key exists and is opaque", key)
      if part not in ancestor:
        if isinstance(key[i + 1], basestring):
          ancestor[part] = Namespace()
        elif isinstance(key[i + 1], numbers.Integral):
          ancestor[part] = []
        else:
          raise KeyError("invalid key type", key[i + 1])
      ancestor = ancestor[part]
    if isinstance(ancestor, Namespace):
      ancestor[key[-1]] = value
    elif isinstance(ancestor, (tuple, list)):
      # lists are hairy; this will fail for out-of-order insertion
      # TODO: implement our own sequence type -_-
      assert len(ancestor) == key[-1]
      ancestor.append(value)

  def __contains__(self, key):
    try:
      self[key]
    except KeyError:
      return False
    else:
      return True

  def __getattr__(self, key):
    if key[0].isupper():
      return self.__dict__[key]
    return self.Data[key]

  def __setattr__(self, key, value):
    if key[0].isupper():
      self.__dict__[key] = value
    self.Data[key] = value

  def __iter__(self):
    return self.Keys()

  def Keys(self):
    return Namespace.DeepKeys(self)

  def Values(self):
    for key in self.Keys():
      yield self[key]

  def Items(self):
    for key in self.Keys():
      yield (key, self[key])

  def __eq__(self, other):
    if not isinstance(other, Namespace):
      return False
    if self is other:
      return True
    for key in self.Data:
      if key not in other.Data or self.Data[key] != other.Data[key]:
        return False
    for key in other.Data:
      if key not in self.Data or other.Data[key] != self.Data[key]:
        return False
    return True

  def __ne__(self, other):
    return not self == other

  def AsDict(self):
    """Convert to an OrderedDict."""
    return odict(self.Data)

  def __repr__(self):
    return "Namespace(%s)" % ", ".join("%s=%s" % (key, repr(value))
                                       for key, value in self.Data.items())

  def __str__(self):
    return "{%s}" % ", ".join("%s: %s" % (key, str(value))
                              for key, value in self.Data.items())

  def Update(self, other):
    """Update `self` with key-value pairs from `other`.

    Args:
      other: Namespace or dict-like
    """
    if isinstance(other, Namespace):
      for key, value in other.Items():
        self[key] = value
    elif hasattr(other, "keys"): # dict
      for key, value in other.items():
        self[key] = value
    else:
      # sequence of pairs
      for key, value in other:
        self[key] = value

  def Extract(self, *keyss):
    """Extract a subnamespace from `self` with the given keys.

    Each of `keyss` is expected to be a string containing space-separated key
    expressions. A key expression can be either a single key or a chain of keys
    for direct access to nested Namespaces. E.g.:

        Namespace(v=2, w=Namespace(x=1, y=Namespace(z=0)))._extract("w.y v")

    would return

        Namespace(w=Namespace(y=Namespace(z=0)), v=2)

    The returned object is a Namespace with the same structure as `self`, except
    that each Namespace contained to it is narrowed to the selected keys.

    Args:
      *keyss: A sequence of strings each containing space-separated key
              expressions.

    Returns:
      The narrowed-down namespace.
    """
    return Namespace((Namespace.DesignatedKey(k), self[key]) for ks in keys for k in ks.split())

  def Get(self, key, default=None):
    try:
      return self[key]
    except KeyError:
      return default

  @staticmethod
  def Flatten(x):
    """Flatten a (possibly nested) Namespace into a list.

    Constructs a list by traversing the tree `x` of nested (lists of) Namespaces
    in order. Descends into Namespaces, tuples and lists. For example:

      ns = Namespace()
      ns.x = [1, 2, Namespace(y=3)]
      ns.z = Namespace(w=[4, 5, (6, 7)])
      assert Namespace.Flatten(ns) == [1, 2, 3, 4, 5, 6, 7]
      assert Namespace.Flatten(ns.x) == [1, 2, 3]
      assert Namespace.Flatten(ns.z.w) == [4, 5, 6, 7]

    Args:
      x: A Namespace, tuple, list or opaque object to flatten.

    Returns:
      The leaf nodes of `x` collected into a list.
    """
    if isinstance(x, (tuple, list)):
      return list(it.chain.from_iterable(map(Namespace.Flatten, x)))
    elif isinstance(x, Namespace):
      return list(it.chain.from_iterable(Namespace.Flatten(x.Data[key]) for key in x.Data))
    else:
      return [x]

  @staticmethod
  def UnflattenLike(xform, yflat):
    """Unflatten a list into a (possibly nested) Namespace.

    This is the inverse of `Flatten`. The structure information is taken from
    the model `xform`. E.g.:

        xflat = Flatten(xform)
        yflat = function_that_likes_lists(xflat)
        yform = UnflattenLike(xform, yflat)

    Args:
      xform: A Namespace tree according to which to structure yflat.
      yflat: A flat list of opaque objects.

    Returns:
      A Namespace tree structured like `xform` with values from `yflat`.
    """
    def _UnflattenLike(xform, yflat):
      if isinstance(xform, (tuple, list)):
        yform = []
        for xelt in xform:
          yelt, yflat = _UnflattenLike(xelt, yflat)
          yform.append(yelt)
      elif isinstance(xform, Namespace):
        yform = Namespace()
        for key in xform.Data:
          yelt, yflat = _UnflattenLike(xform[key], yflat)
          yform[key] = yelt
      else:
        yform, yflat = yflat[0], yflat[1:]
      return yform, yflat
    yform, yflat_leftover = _UnflattenLike(xform, yflat)
    assert not yflat_leftover
    return yform

  @staticmethod
  def Copy(x):
    """Copy a Namespace tree.

    Performs a shallow clone, in the sense that the structure (`Namespace`s,
    lists, tuples) is recreated but the leaf nodes (everything else) are
    copied by reference.

    Args:
      x: The Namespace tree to clone.

    Returns:
      A Namespace tree isomorphic to `x` and with leaf nodes identical to those
      in `x`, but with independent structure.
    """
    return Namespace.UnflattenLike(x, Namespace.Flatten(x))

  @staticmethod
  def FlatCall(fn, tree):
    """Call a list-minded function with objects from a Namespace tree.

    This is a wrapper around `Flatten` and `UnflattenLike` for the common
    case of calling a function that takes and returns lists. The tree is
    flattened into a list, which is passed as an argument to `fn`. `fn`
    returns a list of corresponding outputs, which is unflattened into the
    same structure as `tree` and subsequently returned.

    Args:
      fn: The offending function.
      tree: The Namespace tree to flatten and unflatten.

    Returns:
      The values returned from `fn`, with structure corresponding to `tree`.
    """
    return Namespace.UnflattenLike(tree, fn(Namespace.Flatten(tree)))

  @staticmethod
  def FlatZip(trees, key=None):
    """Zip values from multiple Namespace trees.

    Narrows each of the `trees` to `key` if given, then flattens each tree
    and zips it up.  Example:

        mapping = dict(FlatZip([keys, values]))

    Args:
      trees: the Namespace trees to zip.
      key: optional key to narrow all trees to a common subtree

    Raises:
      ValueError: if the `trees` are not isomorphic and `path` is not given.

    Returns:
      An iterator over tuples with corresponding elements from `trees`.
    """
    trees = list(trees)
    if path:
      trees = [Namespace.Extract(tree, path) for tree in trees]
    if not Namespace.Isomorphic(*trees):
      raise ValueError("Trees not isomorphic")
    return zip(*list(map(Namespace.Flatten, trees)))

  @staticmethod
  def Isomorphic(*trees):
    """Test whether Namespace trees are mutually isomorphic.

    Two trees are isomorphic iff they have the same structure:
      * sequences a and b are isomorphic iff they have the same length and
        all parallel elements a[i] and b[i] are isomorphic.
      * Namespaces a and b are isomorphic iff they have the same set of keys
        and all parallel elements a[key] and b[key] are isomorphic.
      * leaf nodes a and b are isomorphic.

    Args:
      *trees: the trees to compare.

    Returns:
      True if the trees are isomorphic, False otherwise.
    """
    deepkey_lists = [sorted(Namespace.DeepKeys(tree)) for tree in trees]
    if any(len(deepkey_list) != len(deepkey_lists[0]) for deepkey_list in deepkey_lists):
      return False
    for parallel_deepkeys in zip(*deepkey_lists):
      if any(deepkey != parallel_deepkeys[0] for deepkey in parallel_deepkeys):
        return False
    return True

  @staticmethod
  def DeepKeys(tree):
    """Get the keys to all leaf nodes in a Namespace tree.

    Args:
      tree: the tree to traverse.

    Yields:
      keys to each node in the tree, in order, represented by tuples of
           shallow keys.
    """
    if isinstance(tree, (tuple, list)):
      for i, subtree in enumerate(tree):
        for subkey in Namespace.DeepKeys(subtree):
          yield (i,) + subkey
    elif isinstance(tree, Namespace):
      for key, subtree in tree.Data.items():
        for subkey in Namespace.DeepKeys(subtree):
          yield (key,) + subkey
    else:
      yield ()
