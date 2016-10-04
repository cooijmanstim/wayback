import os, tensorflow as tf, numpy as np

from lib.namespace import Namespace as NS
import lib.evaluation as evaluation
import lib.hyperparameters as hyperparameters
import lib.models as models
import lib.training as training
import lib.datasets as datasets
import lib.util as util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("base_output_dir", "/tmp/models", "output directory where models should be stored")
tf.flags.DEFINE_bool("resume", False, "resume training from a checkpoint or delete and restart")
tf.flags.DEFINE_integer("max_step_count", 100000, "max number of training steps")
tf.flags.DEFINE_integer("max_examples", None, "number of examples to train on")
tf.flags.DEFINE_integer("validation_interval", 100, "number of training steps between validations")
tf.flags.DEFINE_integer("tracking_interval", 100, "number of training steps between performance tracking") # fine-grained hits NFS all the time
tf.flags.DEFINE_bool("dump_predictions", False, "dump prediction fragments with every validation")
tf.flags.DEFINE_string("basename", None, "model name prefix")
tf.flags.DEFINE_string("hyperparameters", "", "hyperparameter settings")
tf.flags.DEFINE_string("data_dir", None, "path to data directory; must have" " train/valid/test subdirectories containing wav files")
tf.flags.DEFINE_string("data_type", None, "either wave or bytes")

class StopTraining(Exception):
  pass

def get_model_name(hp):
  fragments = []
  if FLAGS.basename:
    fragments.append(FLAGS.basename)
  fragments.append("sf%d" % hp.sampling_frequency)
  fragments.append("bd%d" % hp.bit_depth)
  fragments.extend([
      hp.layout, hp.cell,
      "s%d" % hp.segment_length,
      "c%d" % hp.chunk_size,
      "p%s" % ",".join(list(map(str, hp.periods))),
      "b%s" % ",".join(list(map(str, hp.boundaries))),
      "l%s" % ",".join(list(map(str, hp.layer_sizes))),
      "u%d" % hp.unroll_layer_count,
      "bn%s" % hp.use_bn,
      "a%s" % hp.activation,
      "bs%d" % hp.batch_size,
      "io%s" % ",".join(list(map(str, hp.io_sizes))),
      "carry%s" % hp.carry,
  ])
  return "_".join(fragments)


def main(argv):
  assert not argv[1:]

  hp = hyperparameters.parse(FLAGS.hyperparameters)

  print "loading data from %s" % FLAGS.data_dir
  dataset = datasets.construct(FLAGS.data_type, directory=FLAGS.data_dir,
                               frequency=hp.sampling_frequency, bit_depth=hp.bit_depth)
  print "done"
  hp.data_dim = dataset.data_dim

  model_name = get_model_name(hp)
  print model_name
  output_dir = os.path.join(FLAGS.base_output_dir, model_name)

  if not FLAGS.resume:
    if tf.gfile.Exists(output_dir):
      tf.gfile.DeleteRecursively(output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  hyperparameters.dump(os.path.join(output_dir, "hyperparameters.yaml"), hp)

  model = models.construct(hp)

  print "constructing graph..."
  global_step = tf.Variable(0, trainable=False, name="global_step")
  trainer = training.Trainer(model, hp=hp, global_step=global_step)
  tf.get_variable_scope().reuse_variables()
  evaluator = evaluation.Evaluator(model, hp=hp)
  print "done"

  best_saver = tf.train.Saver()
  supervisor = tf.train.Supervisor(logdir=output_dir, summary_op=None)
  session = supervisor.PrepareSession()

  tracking = NS(best_loss=None, reset_time=0)

  def track(loss, step):
    if step % FLAGS.tracking_interval == 0:
      if tracking.best_loss is None or loss < tracking.best_loss:
        tracking.best_loss = loss
        tracking.reset_time = step
        best_saver.save(session,
                        os.path.join(os.path.dirname(supervisor.save_path),
                                     "best_%i_%s.ckpt" % (step, loss)),
                        global_step=supervisor.global_step)
      elif step - tracking.reset_time > hp.decay_patience:
        session.run(trainer.tensors.decay_op)
        tracking.reset_time = step

  def maybe_validate(state):
    if state.global_step % FLAGS.validation_interval == 0:
      aggregates = {}
      if FLAGS.dump_predictions:
        # extract final exhats and losses for debugging
        aggregates.update((key, util.LastAggregate()) for key in "seq.final_x final_state.exhats final_state.losses".split())
      values = evaluator.run(examples=dataset.examples.valid, session=session, hp=hp,
                             aggregates=aggregates,
                             # don't spend too much time evaluating
                             max_step_count=FLAGS.validation_interval // 3)
      supervisor.summary_computed(session, tf.Summary(value=values.summaries))
      if FLAGS.dump_predictions:
        np.savez_compressed(os.path.join(os.path.dirname(supervisor.save_path),
                                         "xhats_%i.npz" % state.global_step),
                            # i'm sure we'll get the idea from 100 steps of 10 examples
                            xs=values.seq.final_x[:100, :10],
                            exhats=values.final_state.exhats[:100, :10],
                            losses=values.final_state.losses[:100, :10])
      # track validation loss
      track(values.loss, state.global_step)

  def maybe_stop(_):
    if supervisor.ShouldStop():
      raise StopTraining()

  def before_step_hook(state):
    maybe_validate(state)
    maybe_stop(state)

  def after_step_hook(state, values):
    for summary in values.summaries:
      supervisor.summary_computed(session, summary)
    # decay learning rate based on training loss (we're trying to overfit)
    track(values.loss, state.global_step)

  print "training."
  try:
    trainer.run(examples=dataset.examples.train[:FLAGS.max_examples],
                session=session, hp=hp, max_step_count=FLAGS.max_step_count,
                hooks=NS(step=NS(before=before_step_hook, after=after_step_hook)))
  except StopTraining:
    pass

if __name__ == "__main__":
  tf.app.run()
