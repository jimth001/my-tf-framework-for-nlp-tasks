import tensorflow as tf
from DeepComponents.common_ops import shape_list


def left_to_right_attention_mask(from_tensor, to_tensor, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    Return shape:[1,1,nd,ns]
    """
    assert from_tensor == to_tensor, "from_tensor and to_tensor must be same."
    _, _, nd, ns = shape_list(from_tensor)
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    m = tf.cast(m, dtype)
    m = tf.reshape(m, [1, 1, nd, ns])
    return m


def padding_attention_mask(from_tensor, to_mask, *, dtype):
    """Create 3D attention mask from a 2D tensor mask.

      Args:
        from_tensor: A Tensor of shape [batch_size, heads, from_seq_length, to_seq_length].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

      Returns:
        float Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
      """
    from_shape = shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[2]

    to_shape = shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=dtype)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    mask = tf.expand_dims(mask, axis=1)
    return mask
