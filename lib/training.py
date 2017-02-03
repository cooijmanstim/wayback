import numpy as np, tensorflow as tf, gc
from lib.namespace import Namespace as NS
import lib.tfutil as tfutil
import lib.util as util

class Trainer(object):
  def __init__(self, model, hp, global_step=None):
    self.model = model
    # (+1 because the last chunk is not processed)
    self.segment_length = (self.model.boundary + 1) * hp.chunk_size
    self.tensors = self._make(hp, global_step=global_step)

  def _make(self, hp, global_step=None):
    ts = NS()
    ts.global_step = global_step
    ts.x = tf.placeholder(dtype=tf.int32, name="x")
    ts.seq = self.model.make_training_graph(x=ts.x, length=self.segment_length)
    ts.final_state = ts.seq.final_state
    ts.loss = ts.seq.loss
    ts.error = ts.seq.error

    ts.learning_rate = tf.Variable(hp.initial_learning_rate, dtype=tf.float32,
                                   trainable=False, name="learning_rate")
    ts.decay_op = tf.assign(ts.learning_rate, ts.learning_rate * hp.decay_rate)
    ts.optimizer = tf.train.AdamOptimizer(ts.learning_rate)
    ts.params = tf.trainable_variables()
    print [param.name for param in ts.params]

    ts.gradients = tf.gradients(ts.loss, ts.params)

    loose_params = [param for param, gradient in util.equizip(ts.params, ts.gradients) if gradient is None]
    if loose_params:
      raise ValueError("loose parameters: %s" % " ".join(param.name for param in loose_params))

    # tensorflow fails miserably to compute gradient for these
    for reg_var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      ts.gradients[ts.params.index(reg_var)] += (
          hp.weight_decay * tf.gradients(tf.sqrt(tf.reduce_sum(reg_var**2)), [reg_var])[0])

    ts.clipped_gradients, _ = tf.clip_by_global_norm(ts.gradients, hp.clip_norm)
    ts.training_op = ts.optimizer.apply_gradients(util.equizip(ts.clipped_gradients, ts.params),
                                                  global_step=ts.global_step)

    ts.summaries = [
        tf.summary.scalar("loss_train", ts.loss),
        tf.summary.scalar("error_train", ts.error),
        tf.summary.scalar("learning_rate", ts.learning_rate)
    ]
    for parameter, gradient in util.equizip(ts.params, ts.gradients):
      ts.summaries.append(tf.summary.scalar("meanlogabs_%s"     % parameter.name, tfutil.meanlogabs(parameter)))
      ts.summaries.append(tf.summary.scalar("meanlogabsgrad_%s" % parameter.name, tfutil.meanlogabs(gradient)))

    return ts

  def run(self, session, examples, max_step_count=None, hooks=None, hp=None):
    tensors = self.tensors.Extract("loss error summaries global_step training_op learning_rate final_state.model")
    state = NS(global_step=tf.train.global_step(session, self.tensors.global_step),
               model=self.model.initial_state(hp.batch_size))
    while True:
      for batch in util.batches(examples, hp.batch_size):
        for segment in util.segments(batch, self.segment_length, overlap=hp.chunk_size):
          if max_step_count is not None and state.global_step >= max_step_count:
            return

          hooks.Get("step.before", util.noop)(state)
          x, = util.examples_as_arrays(segment)
          feed_dict = {self.tensors.x: x.T}
          feed_dict.update(self.model.feed_dict(state.model))
          values = tfutil.run(session, tensors, feed_dict=feed_dict)
          state.model = values.final_state.model
          state.global_step = values.global_step
          hooks.Get("step.after", util.noop)(state, values)

          print ("step #%d loss %f error %f learning rate %e" %
                 (values.global_step, values.loss, values.error, values.learning_rate))

          if np.isnan(values.loss):
            raise ValueError("loss has become NaN")
