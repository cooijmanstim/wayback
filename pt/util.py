import contextlib, time

def bouillon(s):
  if s.lowercase() in "1 t true".split():
    return True
  elif s.lowercase() in "0 f false".split():
    return False
  else:
    raise ValueError()

@contextlib.contextmanager
def timing(label):
  start = time.time()
  yield
  end = time.time()
  print("time", label, end - start)

def parse_hp(s):
  d = dict()
  for a in s.split():
    key, value = a.split("=")
    try:
      value = int(value)
    except ValueError:
      try:
        value = float(value)
      except ValueError:
        pass
    d[key] = value
  return d

def serialize_hp(hp, outer_separator=" ", inner_separator="="):
  return outer_separator.join(sorted(["%s%s%s" % (k, inner_separator, v) for k, v in hp.Items()]))

def make_label(config):
  return "%s%s%s" % (config.basename,
                     "_" if config.basename else "",
                     serialize_hp(config.hp, outer_separator="_"))

def equizip(*iterables):
  lists = list(map(list, iterables))
  if not all(len(list) == len(lists[0]) for list in lists):
    raise ValueError("iterables differ in length")
  for elements in zip(*lists):
    yield elements

def batches(examples, batch_size, augment=True, discard_remainder=True):
  # augment: whether to augment examples by random translation
  # discard_remainder: whether to discard last batch if it is smaller than batch_size
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
  offset = 0
  while True:
    batch = examples[offset:offset + batch_size]
    if not batch:
      break
    if len(batch) < batch_size and discard_remainder:
      break
    yield batch
    offset += batch_size

def segments(examples, length, overlap=0, discard_remainder=True):
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

  offset = 0
  while True:
    segments = [example[offset:offset + length] for example in examples]
    if not any(segments):
      break
    if discard_remainder and any(len(segment) < length for segment in segments):
      break
    yield segments
    offset += length - overlap

def examples_as_arrays(examples):
  # NOTE padding would be done here if we ever need it
  return [feature for feature in
          equizip(*[example.render().features for example in examples])]
