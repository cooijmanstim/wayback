import sys, numpy as np, tensorflow as tf
from lib.namespace import Namespace as NS
import lib.tfutil as tfutil
import lib.util as util
from lib.leftover import LEFTOVER

class Evaluator(object):
  def __init__(self, model, hp):
    self.model = model
    self.tensors = self._make(hp)

  def _make(self, unused_hp):
    ts = NS()
    ts.x = tf.placeholder(dtype=tf.int32, name="x")
    ts.seq = self.model.make_evaluation_graph(x=ts.x)
    ts.final_state = ts.seq.final_state
    ts.loss = ts.seq.loss
    ts.error = ts.seq.error
    return ts

  def run(self, session, examples, max_step_count=None, hp=None, aggregates=None):
    aggregates = NS(aggregates or {})
    for key in "loss error".split():
      if key not in aggregates:
        aggregates[key] = util.MeanAggregate()

    tensors = self.tensors.Extract(*[key for key in aggregates.Keys()])
    tensors.Update(self.tensors.Extract("final_state.model"))

    state = NS(step=0, model=self.model.initial_state(hp.batch_size))

    try:
      for batch in util.batches(examples, hp.batch_size):
        for segment in util.segments(batch, 10000, overlap=LEFTOVER):
          if max_step_count is not None and state.step >= max_step_count:
            raise StopIteration()

          x, = util.examples_as_arrays(segment)
          feed_dict = {self.tensors.x: x.T}
          feed_dict.update(self.model.feed_dict(state.model))
          values = tfutil.run(session, tensors, feed_dict=feed_dict)

          for key in aggregates.Keys():
            aggregates[key].add(values[key])

          sys.stderr.write(".")
          state.model = values.final_state.model
          state.step += 1
    except StopIteration:
      pass

    sys.stderr.write("\n")

    values = NS((key, aggregate.value) for key, aggregate in aggregates.Items())

    values.summaries = [tf.Summary.Value(tag="%s_valid" % key, simple_value=values[key])
                        for key in "loss error".split()]
    print "### evaluation loss %6.5f error %6.5f" % (values.loss, values.error)

    if np.isnan(values.loss):
      raise ValueError("loss has become NaN")

    return values
