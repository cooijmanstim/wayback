import glob, os, audioop, wave, numpy as np
import scipy.io.wavfile as wavfile
import util
from holster import H

# examples consist of zero or more "features" (e.g. x and y), each of
# which is a numpy array.  the first axis of each feature is treated
# as the time axis.
class Example(object):
  def __init__(self, features=()):
    if not all(feature.shape[0] == features[0].shape[0] for feature in features):
      raise ValueError("all features must have the same length")
    self._features = features

  @property
  def features(self):
    return self._features

  def __getitem__(self, index):
    return Example(features=tuple(feature[index] for feature in self.features))

  def __len__(self):
    return self.features[0].shape[0]

  def render(self):
    return self

  def with_offset(self, offset):
    return OffsetExample(self.features, offset)

  def map(self, fn):
    return Example(map(fn, self.features))

# some datasets consist of a single long sequence, from which we will
# want to derive many sequences by randomly translating them (with
# wraparound). OffsetExample allows us to represent the translation
# implicitly, deferring the copy to reduce memory use.
class OffsetExample(Example):
  def __init__(self, features=(), offset=0):
    super(OffsetExample, self).__init__(features)
    self.offset = offset

  def __getitem__(self, index):
    if isinstance(index, slice):
      # :-(
      index = list(range(*index.indices(len(self))))
    index = np.asarray(index)
    index += self.offset
    index %= len(self)
    return super(OffsetExample, self).__getitem__(index)

  # ensure raw features are rendered before being accessed
  @property
  def features(self):
    return self.render().features

  # warning: below methods necessarily make copies, and return plain
  # Examples as a result.
  def map(self, fn):
    return self.render().map(fn)

  def render(self):
    return Example([np.roll(feature, -self.offset, axis=0)
                    for feature in self._features])

class Dataset(util.Factory):
  def __init__(self, paths=None, directory=None, **kwargs):
    assert (paths is None) != (directory is None)
    if paths is None:
      paths = H((fold, glob.glob(os.path.join(directory, "%s/*%s" % (fold, self.filename_suffix))))
                for fold in "train valid test".split())
      if not any(paths.Values()):
        # no files found at all, probably not intended
        import pdb; pdb.set_trace()
    self.paths = paths
    self.examples = self.load(self.paths)

  @classmethod
  def make(klass, key, config):
    return super().make(key,
                        paths=config.Get("paths", None),
                        directory=config.Get("directory", None))

  @property
  def data_dim(self):
    raise NotImplementedError()

  def load(self, paths):
    raise NotImplementedError()

  def dump(self, base_path, example):
    raise NotImplementedError()

  def encode(self, x):
    raise NotImplementedError()

  def decode(self, example):
    raise NotImplementedError()

make = Dataset.make

class Bytes(Dataset):
  filename_suffix = ""

  @property
  def data_dim(self):
    # we just deal with any possible byte
    return 256

  def encode(self, bytes):
    return Example([list(map(ord, bytes))])

  def decode(self, example):
    ords, = example.features
    return "".join(map(chr, ords))

  def load(self, paths):
    return H((fold, [self.encode(open(path, "rb").read())
                     for path in fold_paths])
             for fold, fold_paths in paths.Items())

  def dump(self, base_path, example):
    sequence, = example.features
    with open("%s%s" % (base_path, self.filename_suffix), "wb") as outfile:
      outfile.write(self.decode(sequence))

class RestrictedBytes(Bytes):
  filename_suffix = ""
  vocab = ""

  def __init__(self, *args, **kwargs):
    self.bytemap = np.zeros((2**8,), dtype=np.uint8)
    self.bytemap[self.vocab] = np.arange(len(self.vocab))
    super().__init__(*args, **kwargs)

  @property
  def data_dim(self):
    return len(self.vocab)

  def encode(self, bytes):
    return Example([self.bytemap[list(bytes)]])

  def decode(self, example):
    ords, = example.features
    urr = self.bytemap[None, :] == ords[:, None]
    # each category in the example should be in the bytemap
    assert np.allclose(urr.sum(axis=1), 1)
    ords = urr.argmax(axis=1)
    return bytes(ords.astype(np.uint8))

  def load(self, paths):
    return H((fold, [self.encode(open(path, "rb").read())
                     for path in fold_paths])
             for fold, fold_paths in paths.Items())

  def dump(self, base_path, example):
    sequence, = example.features
    with open("%s%s" % (base_path, self.filename_suffix), "wb") as outfile:
      outfile.write(self.decode(sequence))


class Enwik8(RestrictedBytes):
  key = "enwik8"
  filename_suffix = ".txt"
  vocab = list(
    b"""\t\n !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    b"""\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"""
    b"""\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"""
    b"""\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"""
    b"""\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"""
    b"""\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"""
    b"""\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xde"""
    b"""\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xef"""
    b"""\xf0""")


class Linux(RestrictedBytes):
  key = "linux"
  filename_suffix = ""
  vocab = list(
    b"""\t\n\x0c\x14\x1b !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    b"""\x7f"""
    b"""\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"""
    b"""\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"""
    b"""\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"""
    b"""\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"""
    b"""\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"""
    b"""\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"""
    b"""\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"""
    b"""\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff""")


class Wave(Dataset):
  key = "wave"
  filename_suffix = ".wav"

  def __init__(self, paths, frequency, bit_depth, **kwargs):
    self.frequency = frequency
    self.bit_depth = bit_depth
    super().__init__(paths, **kwargs)

  @property
  def data_dim(self):
    return 2 ** self.bit_depth

  def load(self, paths):
    return H((fold, [self.encode(self.load_wavfile(path))
                     for path in fold_paths])
             for fold, fold_paths in paths.Items())

  def dump(self, base_path, example):
    self.dump_wavfile("%s.wav" % base_path, self.decode(example))

  def encode(self, wave):
    return Example([wave])

  def decode(self, example):
    wave, = example.features
    return wave

  def load_wavfile(self, path):
    return load_wavfile(path, bit_depth=self.bit_depth, frequency=self.frequency)

  def dump_wavfile(self, path, sequence):
    dump_wavfile(path, sequence, frequency=self.frequency, bit_depth=self.bit_depth)

def load_wavfile(path, bit_depth, frequency):
  """Load a wav file.

  Resamples the wav file to have sampling frequency `frequency`. The waveform
  is converted to mono, normalized, and its amplitude is discretized into
  `2 ** bit_depth` bins.

  Args:
    path: path to the wav file to load
    bit_depth: resolution of the amplitude discretization, in bits
    frequency: desired sampling frequency

  Returns:
    The waveform as a sequence of categorical integers.
  """
  wav = wave.open(path)
  x = wav.readframes(wav.getnframes())

  # convert to mono
  if wav.getnchannels() > 1:
    x = audioop.tomono(x, wav.getsampwidth(), 0.5, 0.5)

  # convert sampling rate
  x, _ = audioop.ratecv(x, wav.getsampwidth(), 1, wav.getframerate(), frequency, None)

  # center and normalize (done in np.float32 to avoid loss of precision)
  dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[wav.getsampwidth()]
  x = np.frombuffer(x, dtype).astype(np.float32)
  x -= x.mean()
  max_amplitude = abs(x).max()
  # if this happens i'd like to know about it
  assert max_amplitude > 0
  x /= max_amplitude

  x = ulaw(x, mu=bit_depth - 1)
  x = ((2 ** bit_depth - 1) * (x + 1) / 2).round().astype(np.int32)

  return x

def dump_wavfile(path, x, bit_depth, frequency):
  """Dump a wav file.

  Interprets the sequence of integers `x` as a discretized waveform with
  `2 ** bit_depth` amplitude levels and sampling frequency `frequency`,
  and writes the waveform to a mono wav file.

  Args:
    path: path to the wav file to write
    x: the sequence to convert and dump
    bit_depth: resolution of the amplitude discretization, in bits
    frequency: sampling frequency
  """
  x = np.asarray(x, np.float32) / 2 ** bit_depth * 2 - 1
  x = inverse_ulaw(x, mu=bit_depth - 1)
  wavfile.write(path, frequency, x)

def ulaw(x, mu=255):
  return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def inverse_ulaw(y, mu=255):
  return np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
