import os, cStringIO as StringIO, numpy as np, tensorflow as tf

import lib.hyperparameters as hyperparameters
import lib.models as models
import lib.sampling as sampling
import lib.util as util
import lib.wavefile as wavefile

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("sample_duration", 10, "length of sample to generate, in seconds")
tf.flags.DEFINE_float("temperature", 1, "temperature to apply to softmax when sampling")
tf.flags.DEFINE_string("model_hyperparameters", None, "path to model hyperparameter yaml file")
tf.flags.DEFINE_string("model_ckpt", None, "checkpoint from which to load parameter values")
tf.flags.DEFINE_string("base_output_path", None, "prefix for output paths")

def preprocess_primers(examples, hp):
  # maybe augment number of examples to ensure batch norm will work
  min_batch_size = 16
  if len(examples) < min_batch_size:
    k = min_batch_size // len(examples)
    examples.extend(derivation
                    for example in examples
                    for derivation in util.augment_by_random_translations(example, num_examples=k))

  # maybe augment number of time steps to ensure util.segments doesn't discard
  # anything at the ends of the examples. this is done by left-padding the
  # shorter examples with repetitions.
  max_len = max(len(wav) for wav, in examples)
  examples = [[np.pad(wav, [(max_len - len(wav), 0)], mode="wrap")]
              for wav, in examples]

  # time is tight; condition on 3 seconds of the wav files only
  examples_segments = list(util.segments(examples, 3 * hp.sampling_frequency))
  if len(examples_segments) > 2:
    # don't use the first and last segments to avoid silence
    examples_segments = examples_segments[1:-1]
  examples = examples_segments[np.random.choice(len(examples_segments))]

  return examples


def main(argv):
  primer_paths = argv[1:]

  assert FLAGS.model_ckpt
  model_dir = os.path.dirname(FLAGS.model_ckpt)
  if not FLAGS.model_hyperparameters:
    FLAGS.model_hyperparameters = os.path.join(model_dir, "hyperparameters.yaml")
  if not FLAGS.base_output_path:
    FLAGS.base_output_path = model_dir + "/"

  hp = hyperparameters.load(FLAGS.model_hyperparameters)

  assert primer_paths
  dataset = wavefile.Dataset(primer_paths, frequency=hp.sampling_frequency, bit_depth=hp.bit_depth)

  primers = dataset.examples
  primers = preprocess_primers(primers, hp=hp)

  model = models.construct(hp)
  sampler = sampling.Sampler(model, hp=hp)
  saver = tf.train.Saver()
  session = tf.Session("local")
  saver.restore(session, FLAGS.model_ckpt)

  sample_length = hp.sampling_frequency * FLAGS.sample_duration
  xhat = sampler.run(primers=primers, length=sample_length, temperature=FLAGS.temperature, session=session, hp=hp)
  x, = list(map(util.pad, util.equizip(*primers)))

  for i, (p, s) in enumerate(util.equizip(x, xhat)):
    output_path = FLAGS.base_output_path + "temp_%s_%i" % (FLAGS.temperature, i)
    print output_path
    dataset.dump(output_path, [np.concatenate([p, s], axis=0)])

  output_path = FLAGS.base_output_path + "samples.npz"
  print "writing raw sample data to %s" % output_path
  np.savez_compressed(output_path, x=x, xhat=xhat)

if __name__ == "__main__":
  tf.app.run()
