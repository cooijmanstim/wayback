import sys, functools as ft, numpy as np, tensorflow as tf
from lib.namespace import Namespace as NS
import lib.util as util
from lib.leftover import LEFTOVER

class Sampler(object):
  def __init__(self, model, hp):
    self.model = model
    self.tensors = self._make(hp)

  def _make(self, hp):
    ts = NS()
    ts.x = tf.placeholder(dtype=tf.int32, name="x")

    # conditioning graph
    ts.cond = self.model.make_evaluation_graph(x=ts.x)

    # generation graph
    tf.get_variable_scope().reuse_variables()
    ts.initial_xelt = tf.placeholder(dtype=tf.int32, name="initial_xelt", shape=[None])
    ts.length = tf.placeholder(dtype=tf.int32, name="length", shape=[])
    ts.temperature = tf.placeholder(dtype=tf.float32, name="temperature", shape=[])
    ts.sample = self.model.make_sampling_graph(initial_xelt=ts.initial_xelt, length=ts.length, temperature=ts.temperature)

    return ts

  def run(self, session, primers, length, temperature, hp=None):
    batch_size = len(primers)
    # process in segments to avoid tensorflow eating all the memory
    max_segment_length = min(10000, hp.segment_length)

    print "conditioning..."
    segment_length = min(max_segment_length, max(len(primer[0]) for primer in primers))

    state = NS(model=self.model.initial_state(batch_size))
    for segment in util.segments(primers, segment_length, overlap=LEFTOVER):
      x, = util.examples_as_arrays(segment)
      feed_dict = {self.tensors.x: x.T}
      feed_dict.update(self.model.feed_dict(state.model))
      values = NS.FlatCall(ft.partial(session.run, feed_dict=feed_dict),
                           self.tensors.cond.Extract("final_state.model final_xelt"))
      state.model = values.final_state.model
      sys.stderr.write(".")
    sys.stderr.write("\n")

    cond_values = values

    print "sampling..."
    length_left = length + LEFTOVER
    xhats = []
    state = NS(model=cond_values.final_state.model,
               initial_xelt=cond_values.final_xelt)
    while length_left > 0:
      segment_length = min(max_segment_length, length_left)
      length_left -= segment_length

      feed_dict = {self.tensors.initial_xelt: state.initial_xelt,
                   self.tensors.length: segment_length,
                   self.tensors.temperature: temperature}
      feed_dict.update(self.model.feed_dict(state.model))
      sample_values = NS.FlatCall(ft.partial(session.run, feed_dict=feed_dict),
                                  self.tensors.sample.Extract("final_state.model xhat final_xhatelt"))
      state.model = sample_values.final_state.model
      state.initial_xelt = sample_values.final_xhatelt

      xhats.append(sample_values.xhat)
      sys.stderr.write(".")
    sys.stderr.write("\n")

    xhat = np.concatenate(xhats, axis=0)
    return xhat.T
