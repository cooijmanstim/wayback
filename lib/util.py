import collections, logging, numpy as np

def odict(*args, **kwargs):
  return collections.OrderedDict(*args, **kwargs)

def constantly(x):
  return lambda: x

def noop(*unused_args, **unused_kwargs):
  pass

def equizip(*iterables):
  iterators = list(map(iter, iterables))
  for elements in zip(*iterators):
    yield elements
  # double-check that all iterators are exhausted
  for iterator in iterators:
    try:
      _ = next(iterator)
    except StopIteration:
      pass
    else:
      raise ValueError("Sequences must have equal length")

def batches(examples, batch_size, augment=True):
  """Generate randomly chosen batches of examples.

  If the number of examples is not an integer multiple of `batch_size`, the
  remainder is discarded.

  Args:
    examples: iterable
      The examples to choose from.
    batch_size: int
      The desired number of examples per batch.
    augment: bool
      Whether to augment the examples by random translation.

  Raises:
    ValueError: If `len(examples) < batch_size`.

  Yields:
    Subsets of `batch_size` examples.
  """
  examples = list(examples)

  if len(examples) < batch_size and not augment:
    raise ValueError("Not enough examples to fill a batch.")

  if augment:
    # Generate k derivations for each example to ensure we have at least one
    # batch worth of examples. The originals are discarded; if the augmentation
    # is sensible in the first place then using the originals introduces a bias.
    k = int(np.ceil(batch_size / float(len(examples))))
    examples = [example.with_offset(offset)
                for example in examples
                for offset in random_choice(len(example), k)]

  np.random.shuffle(examples)
  for i in range(0, len(examples), batch_size):
    batch = examples[i:i + batch_size]
    if len(batch) < batch_size:
      logging.warning("dropping ragged batch of %d examples", len(batch))
      break
    yield batch

def segments(examples, segment_length, overlap=0, truncate=True):
  """Generate segments from batched sequence data for TBPTT.

  If `truncate` is true, stops as soon as one of the examples runs out, such
  that no padding is needed. Discards the rest.

  Args:
    examples: list of lists of numpy arrays
      The examples to slice up. `examples[i][j]` is the jth feature
      of the ith example.
    segment_length: int
      The desired segment length.
    overlap: int
      The desired number of elements of overlap between consecutive segments.
    truncate: bool
      Whether or not to discard ragged segments where examples vary in length.

  Raises:
    ValueError: If features of an example differ in length or examples differ
    in feature set.

  Yields:
    Slices of examples in the same structure as `examples`. Each segment
    begins where the previous segment left off, except for overlap.
  """
  whichever = 0

  # examples[i] and examples[j] must have the same number of features:
  if not all(len(examples[i].features) == len(examples[whichever].features)
             for i, _ in enumerate(examples)):
    raise ValueError("All examples must have the same set of features.")
  # and their shapes must be the same except for length:
  if not all(examples[i].features[k].shape[1:] == examples[whichever].features[k].shape[1:]
             for i, _ in enumerate(examples)
             for k, _ in enumerate(examples[i])):
    raise ValueError("All examples must have the same set of features.")

  min_length = min(len(example) for example in examples)
  max_length = max(len(example) for example in examples)
  max_offset = min_length - segment_length if truncate else max_length - overlap
  for offset in range(0, max_offset + 1, segment_length - overlap):
    yield [example[offset:offset + segment_length] for example in examples]

def examples_as_arrays(examples):
  return [pad(feature) for feature in
          equizip(*[example.render().features for example in examples])]

def pad(xs):
  """Zero-pad a list of variable-length numpy arrays.

  The arrays are padded along the first axis and stacked into a single array.

  Args:
    xs: The numpy arrays to pad.

  Returns:
    The resulting array.
  """
  y = np.zeros((len(xs), max(map(len, xs))) + xs[0].shape[1:], dtype=xs[0].dtype)
  for i, x in enumerate(xs):
    y[i, :len(x)] = x
  return y

class MeanAggregate(object):
  def __init__(self):
    self.n = 0
    self.v = 0.

  def add(self, x):
    self.v = (self.n * self.v + x) / (self.n + 1)
    self.n += 1

  @property
  def value(self):
    return self.v

class LastAggregate(object):
  def __init__(self):
    self.v = None

  def add(self, x):
    self.v = x

  @property
  def value(self):
    return self.v

def random_choice(n, k=1, rng=np.random):
  # np.random.choice handles this case really poorly, effectively
  # explicitly allocating range(n), shuffling it and taking a slice.
  # the logic here sucks too but it's orders of magnitude more
  # efficient for large n.
  assert n >= k
  if k > n / 2:
    # if k is close to n, the loop below will take longer to complete.
    # but since n is similar to k, allocating range(n) will be okay.
    return rng.choice(n, size=[k], replace=False)
  indices = set()
  while len(indices) < k:
    indices.update(set(rng.randint(n, size=[k - len(indices)])))
  return np.array(list(indices))
