import functools as ft, numpy as np, tensorflow as tf

import lib.cells as cells
from lib.namespace import Namespace as NS
import lib.tfutil as tfutil
import lib.util as util
from lib.leftover import LEFTOVER

def construct(hp):
  activation = dict(tanh=tf.nn.tanh,
                    identity=lambda x: x,
                    elu=tf.nn.elu)[hp.activation]
  cell_implementation = dict(lstm=cells.LSTM,
                             gru=cells.GRU,
                             rnn=cells.RNN,
                             rrnn=cells.RRNN)[hp.cell]
  cells_ = [cell_implementation(layer_size, use_bn=hp.use_bn, activation=activation, scope="cell%i" % i)
            for i, layer_size in enumerate(hp.layer_sizes)]
  model_implementation = dict(stack=Stack, wayback=Wayback)[hp.layout]
  model = model_implementation(cells_=cells_, hp=hp)
  return model

class BaseModel(object):
  """Base class for sequence models.

  The symbolic state for a model is passed around in the form of a Namespace
  tree. This allows arbitrary compositions of models without assumptions on
  the form of their internal state.
  """

  def __init__(self, hp):
    self.hp = hp

  @property
  def period(self):
    """Recurrent period.

    Recurrent models are periodic, but some (e.g. Wayback) take multiple time
    steps to complete a cycle. The period is the number of steps taken to
    complete such a cycle.

    Returns:
      The model's period.
    """
    return 1

  @property
  def boundary(self):
    """Minimum TPBTT segment length."""
    return NotImplementedError()

  def state_placeholders(self):
    """Get the Tensorflow placeholders for the model's states.

    Returns:
      A Namespace tree containing the placeholders.
    """
    return NS.Copy(self._state_placeholders)

  def initial_state(self, batch_size):
    """Get initial values for the model's states.

    Args:
      batch_size: the batch size.

    Returns:
      A Namespace tree containing the values.
    """
    raise NotImplementedError()

  def get_output(self, state):
    """Get model output from the model's states.

    Args:
      state: the model state.

    Returns:
      The model's output.
    """
    raise NotImplementedError()

  def feed_dict(self, state):
    """Construct a feed dict for the model's states.

    Args:
      state: the model state.

    Returns:
      A feed dict mapping each of the model's placeholders to the corresponding
      numerical value in `state`.
    """
    return util.odict(NS.FlatZip([self.state_placeholders(), state]))

  def __call__(self, inputs, state, context=None):
    """Perform a step of the model.

    Args:
      inputs: a `Tensor` denoting the batch of input vectors.
      state: a Namespace tree containing the model's symbolic state.
      context: a `Tensor` denoting context, e.g. for conditioning.

    Returns:
      A tuple of two values: the output and the updated symbolic state.
    """
    raise NotImplementedError()

  def _make_sequence_graph(self, **kwargs):
    """See the module-level `_make_sequence_graph`."""
    if kwargs.get("model_state", None) is None:
      kwargs["model_state"] = self.state_placeholders()
    def transition(input_, state, context=None):
      state = self(input_, state, context=context)
      h = self.get_output(state)
      return h, state
    return _make_sequence_graph(transition=transition, **kwargs)

  def make_training_graph(self, x, length=None, context=None, model_state=None):
    """Make a graph to train the model by teacher-forcing.

    `x` is processed elementwise. At step `i`, the model receives the `i`th element as input, and
    its output is used to predict the `i + 1`th element.

    The last element is not processed, as there would be no further element available to compare
    against and compute loss. To ensure all data is processed during TBPTT, segments `x` fed into
    successive computations of the graph should overlap by 1.

    Args:
      x: Sequence of integer (categorical) inputs, shaped [time, batch].
      length: Optional length of sequence. Inferred from `x` if possible.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(x=x, model_state=model_state,
                                     length=length, context=context,
                                     back_prop=True, hp=self.hp)

  def make_evaluation_graph(self, x, length=None, context=None,
                            model_state=None):
    """Make a graph to evaluate the model.

    `x` is processed elementwise. At step `i`, the model receives the `i`th element as input, and
    its output is used to predict the `i + 1`th element.

    The last element is not processed, as there would be no further element available to compare
    against and compute loss. To ensure all data is processed during TBPTT, segments `x` fed into
    successive computations of the graph should overlap by 1.

    Args:
      x: Sequence of integer (categorical) inputs, shaped [time, batch].
      length: Optional length of sequence. Inferred from `x` if possible.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(x=x, model_state=model_state,
                                     length=length, context=context, hp=self.hp)

  def make_sampling_graph(self, initial_xelt, length, context=None, model_state=None, temperature=1.0):
    """Make a graph to sample from the model.

    The graph generates a sequence `xhat` one element at a time. At the first step, the model
    receives `initial_xelt` as input, and generates an element to follow it. The generated element
    is used as input during the next time step. This process is repeated until a sequence of the
    desired length has been generated.

    Args:
      initial_xelt: Initial model input, shaped [batch].
      length: Desired length of generated sequence.
      context: Optional Tensor denoting context, shaped [batch, ?].
      model_state: Initial state of the model.
      temperature: Softmax temperature to use for sampling.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    return self._make_sequence_graph(initial_xelt=initial_xelt,
                                     model_state=model_state, length=length,
                                     context=context, temperature=temperature,
                                     hp=self.hp)

class Stack(BaseModel):
  def __init__(self, cells_, hp):
    """Initialize a `Stack` instance.

    Args:
      cells_: recurrent transition cells, from bottom to top.
      hp: model hyperparameters.
    """
    super(Stack, self).__init__(hp)
    self.cells = list(cells_)
    self._state_placeholders = NS(cells=[cell.state_placeholders for cell in self.cells])

  @property
  def boundary(self):
    return max(self.hp.boundaries)

  def initial_state(self, batch_size):
    return NS(cells=[cell.initial_state(batch_size) for cell in self.cells])

  def get_output(self, state):
    return tf.concat(1, [cell.get_output(cell_state) for cell, cell_state in zip(self.cells, state.cells)])

  def __call__(self, x, state, context=None):
    state = NS.Copy(state)
    for i, _ in enumerate(self.cells):
      cell_inputs = []
      if i == 0:
        cell_inputs.append(x)
      if context is not None and i == len(self.cells) - 1:
        cell_inputs.append(context)
      if self.hp.vskip:
        # feed in state of all other layers
        cell_inputs.extend(self.cells[j].get_output(state.cells[j])
                           for j in range(len(self.cells)) if j != i)
      else:
        # feed in state of layer below
        if i > 0:
          cell_inputs.append(self.cells[i - 1].get_output(state.cells[i - 1]))
      state.cells[i] = self.cells[i].transition(cell_inputs, state.cells[i], scope="cell%i" % i)
    return state

class Wayback(BaseModel):
  def __init__(self, cells_, hp):
    """Initialize a `Wayback` instance.

    The following hyperparameters are specific to this model:
      periods: update interval of each layer, from top to bottom. periods[0]
          should be 1 unless you're trying to skip inputs.
      unroll_layer_count: number of upper layers to unroll. Unrolling allows
          for gradient truncation on the levels below.
      carry: whether to carry over each cell's state from one cycle to the next
          or break the chain and compute new initial states based on the state
          of the cell above.

    Args:
      cells_: recurrent transition cells, from top to bottom.
      hp: model hyperparameters.

    Raises:
      ValueError: If the number of cells and the number of periods differ.
    """
    super(Wayback, self).__init__(hp)

    if len(self.hp.periods) != len(cells_):
      raise ValueError("must specify one period for each cell")
    if len(self.hp.boundaries) != len(cells_):
      raise ValueError("must specify one boundary for each cell")
    self.cells = list(cells_)

    cutoff = len(cells_) - self.hp.unroll_layer_count
    self.inner_indices = list(range(cutoff))
    self.outer_indices = list(range(cutoff, len(cells_)))
    self.inner_slice = slice(cutoff)
    self.outer_slice = slice(cutoff, len(cells_))

    self._state_placeholders = NS(time=tf.placeholder(dtype=tf.int32, name="time"),
                                  cells=[cell.state_placeholders for cell in self.cells])

  @property
  def period(self):
    return int(np.prod(self.hp.periods))

  @property
  def boundary(self):
    return self.hp.boundaries[-1] * self.period

  def initial_state(self, batch_size):
    return NS(time=0, cells=[cell.initial_state(batch_size) for cell in self.cells])

  def get_output(self, state):
    return tf.concat(1, [cell.get_output(cell_state) for cell, cell_state in zip(self.cells, state.cells)])

  def __call__(self, x, state, context=None):
    # construct the usual graph without unrolling
    state = NS.Copy(state)
    state.cells = Wayback.transition(state.time, state.cells, self.cells,
                                     below=x, above=context, hp=self.hp,
                                     symbolic=True)
    state.time += 1
    state.time %= self.period
    return state

  @staticmethod
  def transition(time, cell_states, cells_, subset=None,
                 below=None, above=None, hp=None, symbolic=True):
    """Perform one Wayback transition.

    This function updates `cell_states` according to the wayback connection
    pattern. Note that although `subset` selects a subset of cells to update,
    this function will reach outside that subset to properly condition the
    states within and potentially to disconnect gradient on cells below those in
    `subset`.

    Args:
      time: model time as kept by model state. Must be a Python int if
          `symbolic` is true.
      cell_states: list of cell states as kept by model state.
      cells_: list of cell instances.
      subset: indices of cells to update
      below: Tensor context from below.
      above: Tensor context from above.
      hp: model hyperparameters.
      symbolic: whether the transition occurs inside a dynamic loop or not.

    Returns:
      Updated cell states. Updating `time` is the caller's responsibility.
    """
    def _is_due(i):
      countdown = time % np.prod(hp.periods[:i + 1])
      return tf.equal(countdown, 0) if symbolic else countdown == 0

    subset = list(range(len(cells_))) if subset is None else subset
    for i in reversed(sorted(subset)):
      is_top = i == len(cells_) - 1
      is_bottom = i == 0

      cell_inputs = []
      if is_bottom and below is not None:
        cell_inputs.append(below)
      if is_top and above is not None:
        cell_inputs.append(above)

      if hp.vskip:
        # feed in states of all other layers
        cell_inputs.extend(cells_[j].get_output(cell_states[j])
                           for j in range(len(cells_)) if j != i)
      else:
        # feed in state of layers below and above
        if not is_bottom:
          cell_inputs.append(cells_[i - 1].get_output(cell_states[i - 1]))
        if not is_top:
          cell_inputs.append(cells_[i + 1].get_output(cell_states[i + 1]))

      # NOTE: Branch functions passed to `cond` don't get called until way after
      # we're out of this loop. That means we need to be careful not to pass in
      # a closure with a loop variable.

      cell_state = cell_states[i]
      carry_context = above if is_top else cells_[i + 1].get_output(cell_states[i + 1])
      if not hp.carry and carry_context is not None:
        # start every cycle with a new initial state determined from state above
        def _reinit_cell(cell, context, scope):
          return [tfutil.layer([context], output_dim=size, use_bn=hp.use_bn, scope="%s_%i" % (scope, j))
                  for j, size in enumerate(cell.state_size)]
        reinit_cell = ft.partial(_reinit_cell, cell=cells_[i], context=carry_context, scope="reinit_%i" % i)
        preserve_cell = util.constantly(cell_state)

        # reinitialize cell[i] if cell above was just updated
        if symbolic:
          cell_state = tfutil.cond(_is_due(i + 1), reinit_cell, preserve_cell, prototype=cell_state)
        else:
          if _is_due(i + 1):
            cell_state = reinit_cell()

      update_cell = ft.partial(cells_[i].transition, cell_inputs, cell_state)
      preserve_cell = util.constantly(cell_state)

      if symbolic:
        if is_bottom:
          # skip the cond; bottom layer updates each step
          cell_states[i] = update_cell()
        else:
          cell_states[i] = tfutil.cond(_is_due(i), update_cell, preserve_cell, prototype=cell_state)
      else:
        if _is_due(i):
          cell_states[i] = update_cell()

    return cell_states

  def _make_sequence_graph(self, **kwargs):
    """Create a (partially unrolled) sequence graph.

    Where possible, this method calls `BaseModel._make_sequence_graph` to
    construct a simple graph with a single while loop.

    If `back_prop` is true and the model is configured for partial unrolling,
    this method dispatches to `Wayback._make_sequence_graph_with_unroll`. In
    that case, `length` must be an int.

    Args:
      **kwargs: passed onto `Wayback._make_sequence_graph_with_unroll` or
                `Wayback._make_sequence_graph`.

    Returns:
      A Namespace containing relevant symbolic variables.
    """
    if kwargs.get("back_prop", False) and self.outer_indices:
      return self._make_sequence_graph_with_unroll(**kwargs)
    else:
      return super(Wayback, self)._make_sequence_graph(**kwargs)

  def _make_sequence_graph_with_unroll(self, model_state=None, x=None,
                                       initial_xelt=None, context=None,
                                       length=None, temperature=1.0, hp=None,
                                       back_prop=False):
    """Create a sequence graph by unrolling upper layers.

    This method is similar to `_make_sequence_graph`, except that `length` must be provided. The
    resulting graph behaves in the same way as that constructed by `_make_sequence_graph`, except
    that the upper layers are outside of the while loop and so the gradient can actually be
    truncated between runs of lower layers.

    If `x` is given, the graph processes the sequence `x` one element at a time.  At step `i`, the
    model receives the `i`th element as input, and its output is used to predict the `i + 1`th
    element.

    The last element is not processed, as there would be no further element available to compare
    against and compute loss. To ensure all data is processed during TBPTT, segments `x` fed into
    successive computations of the graph should overlap by 1.

    If `x` is not given, `initial_xelt` must be given as the first input to the model.  Further
    elements are constructed from the model's predictions.

    Args:
      model_state: initial state of the model.
      x: Sequence of integer (categorical) inputs. Not needed if sampling.
          Axes [time, batch].
      initial_xelt: When sampling, x is not given; initial_xelt specifies
          the input x[0] to the first timestep.
      context: a `Tensor` denoting context, e.g. for conditioning.
          Axes [batch, features].
      length: Optional length of sequence. Inferred from `x` if possible.
      temperature: Softmax temperature to use for sampling.
      hp: Model hyperparameters.
      back_prop: Whether the graph will be backpropagated through.

    Raises:
      ValueError: if `length` is not an int.

    Returns:
      Namespace containing relevant symbolic variables.
    """
    if length is None or not isinstance(length, int):
      raise ValueError("For partial unrolling, length must be known at graph construction time.")

    if model_state is None:
      model_state = self.state_placeholders()

    state = NS(model=model_state, inner_initial_xelt=initial_xelt, xhats=[], losses=[], errors=[])

    # i suspect ugly gradient biases may occur if gradients are truncated
    # somewhere halfway through the cycle. ensure we start at a cycle boundary.
    state.model.time = tfutil.assertion(state.model.time,
                                        tf.equal(state.model.time, 0),
                                        [state.model.time],
                                        name="outer_alignment_assertion")
    # ensure we end at a cycle boundary too.
    assert (length - LEFTOVER) % self.period == 0

    inner_period = int(np.prod(hp.periods[:self.outer_indices[0] + 1]))

    # hp.boundaries specifies truncation boundaries relative to the end of the sequence and in terms
    # of each layer's own steps; translate this to be relative to the beginning of the sequence and
    # in terms of sequence elements. note that due to the dynamic unrolling of the inner graph, the
    # inner layers necessarily get truncated at the topmost inner layer's boundary.
    boundaries = [length - 1 - hp.boundaries[i] * int(np.prod(hp.periods[:i + 1]))
                  for i in range(len(hp.periods))]
    assert all(0 <= boundary and boundary < length - LEFTOVER for boundary in boundaries)
    assert boundaries == list(reversed(sorted(boundaries)))

    print "length %s periods %s boundaries %s %s inner period %s" % (length, hp.periods, hp.boundaries, boundaries, inner_period)

    outer_step_count = length // inner_period
    for outer_time in range(outer_step_count):
      if outer_time > 0:
        tf.get_variable_scope().reuse_variables()

      # update outer layers (wrap in seq scope to be consistent with the fully
      # symbolic version of this graph)
      with tf.variable_scope("seq"):
        # truncate gradient (only effective on outer layers)
        for i in range(len(self.cells)):
          if outer_time * inner_period <= boundaries[i]:
            state.model.cells[i] = list(map(tf.stop_gradient, state.model.cells[i]))

        state.model.cells = Wayback.transition(
            outer_time * inner_period, state.model.cells, self.cells,
            below=None, above=context, subset=self.outer_indices, hp=hp,
            symbolic=False)

      # run inner layers on subsequence
      if x is None:
        inner_x = None
      else:
        start = inner_period *  outer_time
        stop  = inner_period * (outer_time + 1) + LEFTOVER
        inner_x = x[start:stop, :]

      # grab a copy of the outer states. they will not be updated in the inner
      # loop, so we can put back the copy after the inner loop completes.
      # this avoids the gradient truncation due to calling `while_loop` with
      # `back_prop=False`.
      outer_cell_states = NS.Copy(state.model.cells[self.outer_slice])

      def _inner_transition(input_, state, context=None):
        assert not context
        state.cells = Wayback.transition(
            state.time, state.cells, self.cells, below=input_, above=None,
            subset=self.inner_indices, hp=hp, symbolic=True)
        state.time += 1
        state.time %= self.period
        h = self.get_output(state)
        return h, state

      inner_back_prop = back_prop and outer_time * inner_period >= boundaries[self.inner_indices[-1]]
      inner_ts = _make_sequence_graph(
          transition=_inner_transition, model_state=state.model,
          x=inner_x, initial_xelt=state.inner_initial_xelt,
          temperature=temperature, hp=hp,
          back_prop=inner_back_prop)

      state.model = inner_ts.final_state.model
      state.inner_initial_xelt = inner_ts.final_xelt if x is not None else inner_ts.final_xhatelt
      state.final_xhatelt = inner_ts.final_xhatelt
      if x is not None:
        state.final_x = inner_x
        state.final_xelt = inner_ts.final_xelt
        # track only losses and errors after the boundary to avoid bypassing the truncation boundary.
        if inner_back_prop:
          state.losses.append(inner_ts.loss)
          state.errors.append(inner_ts.error)
      state.xhats.append(inner_ts.xhat)

      # restore static outer states
      state.model.cells[self.outer_slice] = outer_cell_states

      # double check alignment to be safe
      state.model.time = tfutil.assertion(state.model.time,
                                          tf.equal(state.model.time % inner_period, 0),
                                          [state.model.time, tf.shape(inner_x)],
                                          name="inner_alignment_assertion")

    ts = NS()
    ts.xhat = tf.concat(0, state.xhats)
    ts.final_xhatelt = state.final_xhatelt
    ts.final_state = state
    if x is not None:
      ts.final_x = state.final_x
      ts.final_xelt = state.final_xelt
      # inner means are all on the same sample size, so taking their mean is valid
      ts.loss = tf.reduce_mean(state.losses)
      ts.error = tf.reduce_mean(state.errors)
    return ts

def _make_sequence_graph(transition=None, model_state=None, x=None,
                         initial_xelt=None, context=None, length=None,
                         temperature=1.0, hp=None, back_prop=False):
  """Construct the graph to process a sequence of categorical integers.

  If `x` is given, the graph processes the sequence `x` one element at a time.  At step `i`, the
  model receives the `i`th element as input, and its output is used to predict the `i + 1`th
  element.

  The last element is not processed, as there would be no further element available to compare
  against and compute loss. To ensure all data is processed during TBPTT, segments `x` fed into
  successive computations of the graph should overlap by 1.

  If `x` is not given, `initial_xelt` must be given as the first input to the model.  Further
  elements are constructed from the model's predictions.

  Args:
    transition: model transition function mapping (xelt, model_state,
        context) to (output, new_model_state).
    model_state: initial state of the model.
    x: Sequence of integer (categorical) inputs. Not needed if sampling.
        Axes [time, batch].
    initial_xelt: When sampling, x is not given; initial_xelt specifies
        the input x[0] to the first timestep.
    context: a `Tensor` denoting context, e.g. for conditioning.
    length: Optional length of sequence. Inferred from `x` if possible.
    temperature: Softmax temperature to use for sampling.
    hp: Model hyperparameters.
    back_prop: Whether the graph will be backpropagated through.

  Returns:
    Namespace containing relevant symbolic variables.
  """
  with tf.variable_scope("seq") as scope:
    # if the caching device is not set explicitly, set it such that the
    # variables for the RNN are all cached locally.
    if scope.caching_device is None:
      scope.set_caching_device(lambda op: op.device)

    if length is None:
      length = tf.shape(x)[0]

    def _make_ta(name, **kwargs):
      # infer_shape=False because it is too strict; it considers unknown
      # dimensions to be incompatible with anything else. Effectively that
      # requires all shapes to be fully defined at graph construction time.
      return tf.TensorArray(tensor_array_name=name, infer_shape=False, **kwargs)

    state = NS(i=tf.constant(0), model=model_state)

    state.xhats = _make_ta("xhats", dtype=tf.int32, size=length, clear_after_read=False)
    state.xhats = state.xhats.write(0, initial_xelt if x is None else x[0, :])

    state.exhats = _make_ta("exhats", dtype=tf.float32, size=length - LEFTOVER)

    if x is not None:
      state.losses = _make_ta("losses", dtype=tf.float32, size=length - LEFTOVER)
      state.errors = _make_ta("errors", dtype=tf.bool,    size=length - LEFTOVER)

    state = tfutil.while_loop(cond=lambda state: state.i < length - LEFTOVER,
                              body=ft.partial(make_transition_graph,
                                              transition=transition, x=x, context=context,
                                              temperature=temperature, hp=hp),
                              loop_vars=state,
                              back_prop=back_prop)

    # pack TensorArrays
    for key in "exhats xhats losses errors".split():
      if key in state:
        state[key] = state[key].pack()

    ts = NS()
    ts.final_state = state
    ts.xhat = state.xhats[1:, :]
    ts.final_xhatelt = state.xhats[length - 1, :]
    if x is not None:
      ts.loss = tf.reduce_mean(state.losses)
      ts.error = tf.reduce_mean(tf.to_float(state.errors))
      ts.final_x = x
      # expose the final, unprocessed element of x for convenience
      ts.final_xelt = x[length - 1, :]
    return ts

def make_transition_graph(state, transition, x=None, context=None,
                          temperature=1.0, hp=None):
  """Make the graph that processes a single sequence element.

  Args:
    state: `_make_sequence_graph` loop state.
    transition: Model transition function mapping (xelt, model_state,
        context) to (output, new_model_state).
    x: Sequence of integer (categorical) inputs. Axes [time, batch].
    context: Optional Tensor denoting context, shaped [batch, ?].
    temperature: Softmax temperature to use for sampling.
    hp: Model hyperparameters.

  Returns:
    Updated loop state.
  """
  state = NS.Copy(state)

  xelt = tfutil.shaped_one_hot(state.xhats.read(state.i) if x is None else x[state.i, :],
                               [None, hp.data_dim])
  embedding = tfutil.layers([xelt], sizes=hp.io_sizes, use_bn=hp.use_bn)
  h, state.model = transition(embedding, state.model, context=context)

  # predict the next elt
  with tf.variable_scope("xhat") as scope:
    embedding = tfutil.layers([h], sizes=hp.io_sizes, use_bn=hp.use_bn)
    exhat = tfutil.project(embedding, output_dim=hp.data_dim)
    xhat = tfutil.sample(exhat, temperature)
    state.xhats = state.xhats.write(state.i + LEFTOVER, xhat)

  if x is not None:
    target = tfutil.shaped_one_hot(x[state.i + 1], [None, hp.data_dim])
    state.losses = state.losses.write(state.i, tf.nn.softmax_cross_entropy_with_logits(exhat, target))
    state.errors = state.errors.write(state.i, tf.not_equal(tf.nn.top_k(exhat)[1], tf.nn.top_k(target)[1]))
    state.exhats = state.exhats.write(state.i, exhat)

  state.i += 1
  return state
