
import tensorflow as tf


def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.
  Args:
    logits: Tensor of shape [batch_size, ?, ?, ?, vocab_size].
    labels: Tensor of shape [batch_size, ?, ?, ?].
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the Gaussian
      distribution.
    gaussian: Uses a Gaussian distribution for label smoothing
  Returns:
    Tensor of shape [batch_size, ?, ?, ?].
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing



def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return tf.to_float(tf.not_equal(labels, 0))


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y


def pad_with_zeros(logits, labels):
  """Pad labels on the length dimension to match logits length."""
  with tf.name_scope("pad_with_zeros", values=[logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return logits, labels



class FactoredTensor(object):
  """A concise factored representation of Tensor as two tensors.
  This class represents the tensor tf.matmul(a, b, transpose_b=True)
  by storing the values of Tensors a and b.
  The reason for this is that the product may be too big to fully realize at
  once, so it can be realized a part at a time.
  "a" may have extra leading dimensions, in which case they are flattened out
  before computing the matrix product, then re-expanded afterwards.
  """

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def to_tensor(self):
    """Convert to Tensor."""
    a_shape = shape_list(self.a)
    b_shape = shape_list(self.b)
    inner_dim = b_shape[1]
    result_dim = b_shape[0]
    flat_a = tf.reshape(self.a, [-1, inner_dim])
    product = tf.matmul(flat_a, self.b, transpose_b=True)
    product_shape = a_shape[:-1] + [result_dim]
    product = tf.reshape(product, product_shape)
    product.set_shape(self.a.get_shape().as_list()[:-1] +
                      [self.b.get_shape()[0]])
    return product



def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.
  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.
  Returns:
    a list of num_splits Tensors.
  """
  size = shape_list(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


def smoothing_cross_entropy_factored(a, b, labels, confidence):
  """Memory-efficient computation of smoothing cross-entropy.
  Avoids realizing the entire logits matrix at once.
  Args:
    a: a Tensor with shape [batch, inner_dim]
    b: a Tensor with shape [vocab_size, inner_dim]
    labels: an integer Tensor with shape [batch]
    confidence: a float
  Returns:
    A Tensor with shape [batch]
  """
  num_splits = 16
  vocab_size = shape_list(b)[0]
  labels = approximate_split(labels, num_splits)
  a = approximate_split(a, num_splits)
  parts = []
  for part in range(num_splits):
    with tf.control_dependencies(parts[-1:]):
      logits = tf.matmul(a[part], b, transpose_b=True)
      parts.append(
          smoothing_cross_entropy(logits, labels[part], vocab_size, confidence))
  return tf.concat(parts, 0)



def padded_cross_entropy_factored(factored_logits,
                                  labels,
                                  label_smoothing,
                                  weights_fn=weights_nonzero,
                                  reduce_sum=True):
  """Memory-efficient computation of smoothing cross-entropy.
  Avoids realizing the entire logits matrix at once.
  Args:
    factored_logits: a `FactoredTensor` representing a Tensor
       with shape `[batch, timesteps, vocab_size]`.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.
  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.
  """
  a = factored_logits.a
  b = factored_logits.b
  confidence = 1.0 - label_smoothing
  with tf.name_scope("padded_cross_entropy_factored", values=[a, b, labels]):
    labels_flat = tf.reshape(labels, [-1])
    a_flat = tf.reshape(a, [-1, shape_list(b)[1]])
    xent = smoothing_cross_entropy_factored(a_flat, b, labels_flat,
                                            tf.convert_to_tensor(confidence))
    xent = tf.reshape(xent, shape_list(labels))
    weights = weights_fn(labels)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True,
                         cutoff=0.0,
                         gaussian=False):
  """Compute cross-entropy assuming 0s are padding.
  Computes a loss numerator (the sum of losses), and loss denominator
  (the number of non-padding tokens).
  Args:
    logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
      optionally a FactoredTensor.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.
    cutoff: a float, at which point to have no loss.
    gaussian: If true, use a Gaussian distribution for label smoothing
  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.
  Raises:
    ValueError: in case of unsupported argument types.
  """
  if isinstance(logits, FactoredTensor):
    if gaussian:
      raise ValueError("Factored padded cross entropy with Gaussian smoothing "
                       "is not implemented yet.")
    return padded_cross_entropy_factored(
        logits,
        labels,
        label_smoothing,
        weights_fn=weights_fn,
        reduce_sum=reduce_sum)
  confidence = 1.0 - label_smoothing
  logits_shape = shape_list(logits)
  vocab_size = logits_shape[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    if len(logits_shape) == 2:
      # Deal with the case where we did not insert extra dimensions due to
      # TPU issues.  No pad-to-same-length happens in this case.
      # TODO(noam): remove this logic once TPU can handle extra dimensions.
      labels = tf.reshape(labels, [-1])
    else:
      logits, labels = pad_with_zeros(logits, labels)
    logits = tf.reshape(
        logits,
        shape_list(labels) + [vocab_size],
        name="padded_cross_entropy_size_check")
    logits = tf.cast(logits, tf.float32)
    xent = smoothing_cross_entropy(
        logits, labels, vocab_size, confidence, gaussian=gaussian)
    weights = weights_fn(labels)
    if cutoff > 0.0:
      xent = tf.nn.relu(xent - cutoff)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)

