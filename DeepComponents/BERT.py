import tensorflow as tf
from DeepComponents.TransformerBlock import shape_list, positions_for, block, norm

def model(*, hparams, X, src_seq_mask, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))
        # Transformer
        presents = []
        for layer in range(hparams.n_layer):
            h, present = block(h, 'h%d' % layer, past=None, hparams=hparams, src_seq_mask=src_seq_mask)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')
        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
