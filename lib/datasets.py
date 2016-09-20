import glob, os, audioop, wave, numpy as np, tensorflow as tf
import scipy.io.wavfile as wavfile
from lib.namespace import Namespace as NS

def construct(data_type, paths=None, directory=None, **kwargs):
  klass = dict(wave=Wave, bytes=Bytes, enwik8=Enwik8, linux=Linux)[data_type]
  return klass(paths=paths, directory=directory, **kwargs)

class Dataset(object):
  def __init__(self, paths=None, directory=None, **kwargs):
    assert (paths is None) != (directory is None)
    if paths is None:
      paths = NS((fold, glob.glob(os.path.join(directory, "%s/*%s" % (fold, self.filename_suffix))))
                 for fold in "train valid test".split())
      if not NS.Flatten(paths):
        # no files found at all, probably not intended
        import pdb; pdb.set_trace()
    self.paths = paths
    self.examples = self.load(self.paths)

  @property
  def data_dim(self):
    raise NotImplementedError()

  def dump(self, base_path, example):
    raise NotImplementedError()

class Bytes(Dataset):
  filename_suffix = ""

  @property
  def data_dim(self):
    # we just deal with any possible byte
    return 256

  def load(self, paths):
    return NS.UnflattenLike(paths,
                            [[np.array(list(map(ord, open(path, "rb").read())), dtype=np.int32)]
                             for path in NS.Flatten(paths)])

  def dump(self, base_path, example):
    sequence, = example
    with open("%s%s" % (base_path, self.filename_suffix), "wb") as outfile:
      outfile.write(sequence)

class RestrictedBytes(Bytes):
  filename_suffix = ""
  vocab = ""

  @property
  def data_dim(self):
    return len(self.vocab)

  def load(self, paths):
    bytemap = np.zeros((2**8,), dtype=np.int32)
    bytemap[list(map(ord, self.vocab))] = np.arange(len(self.vocab))
    return NS.UnflattenLike(paths,
                            [[bytemap[list(map(ord, open(path, "rb").read()))]]
                             for path in NS.Flatten(paths)])

  def dump(self, base_path, example):
    sequence, = example
    with open("%s%s" % (base_path, self.filename_suffix), "wb") as outfile:
      outfile.write(sequence)


class Enwik8(RestrictedBytes):
  filename_suffix = ".txt"
  vocab = list(
    """\t\n !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    """\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"""
    """\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"""
    """\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"""
    """\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"""
    """\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"""
    """\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xde"""
    """\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xef"""
    """\xf0""")


class Linux(RestrictedBytes):
  filename_suffix = ""
  vocab = list(
    """\t\n\x0c\x14\x1b !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    """\x7f"""
    """\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"""
    """\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"""
    """\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"""
    """\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"""
    """\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"""
    """\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"""
    """\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"""
    """\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff""")


class Wave(Dataset):
  filename_suffix = ".wav"

  def __init__(self, paths, frequency, bit_depth, **kwargs):
    """Initialize a Wav instance.

    The wav files referenced by `paths` will be made available under the
    `examples` attribute, in the same Namespace tree structure.

    Args:
      paths: Namespace tree with wav file paths
      frequency: desired sampling frequency
      bit_depth: desired amplitude resolution in bits
    """
    self.frequency = frequency
    self.bit_depth = bit_depth
    super(Wave, self).__init__(paths, **kwargs)

  @property
  def data_dim(self):
    """Number of classes."""
    return 2 ** self.bit_depth

  def load(self, paths):
    """Load data.

    Args:
      paths: Namespace tree with wav file paths

    Returns:
      Isomorphic Namespace tree with waveform examples.
    """
    return NS.UnflattenLike(paths, [[self.load_wavfile(path)] for path in NS.Flatten(paths)])

  def dump(self, base_path, example):
    """Dump a single example.

    Args:
      base_path: the path of the file to write (without extension)
      example: the waveform example to write
    """
    sequence, = example
    self.dump_wavfile("%s.wav" % base_path, sequence)

  def load_wavfile(self, path):
    """Load a single wav file.

    This is like `load_wavfile` but specifies the frequency and bit_depth.

    Args:
      path: path to the wav file to load

    Returns:
      The waveform as a sequence of categorical integers.
    """
    return load_wavfile(path, bit_depth=self.bit_depth, frequency=self.frequency)

  def dump_wavfile(self, path, sequence):
    """Dump a single wav file.

    This is like `dump_wavfile` but specifies the frequency and bit_depth.

    Args:
      path: where to dump the wav file.
      sequence: the sequence to dump.
    """
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
