import tensorflow as tf
from DeepComponents.TransformerBlock import positions_for, block, norm, past_shape
from DeepComponents.common_ops import gather_2d

class Encoder():
    def __init__(self,scope,hparam):
        if scope is None:
            self.scope='encoder'
        else:
            self.scope=scope
        self.hparams=hparam

    #for uni_sls
    def encode_which_outputs_all_layer_h(self, X, h_len, past=None, scope='encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                # Transformer
                wpe = tf.get_variable('wpe', [self.hparams.n_ctx, self.hparams.n_embd],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
                wte = tf.get_variable('wte', [self.hparams.n_vocab, self.hparams.n_embd],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                past_length = 0 if past is None else tf.shape(past)[-2]
                h = tf.gather(wte, X, name='gggggg1') + tf.gather(wpe, positions_for(X, past_length), name='ggggggg2')
                presents = []
                pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
                assert len(pasts) == self.hparams.n_layer
                all_h = []
                final_id = h_len - 1
                for layer, past_one in enumerate(pasts):
                    h, present = block(h, 'h%d' % layer, past=past_one, hparams=self.hparams)
                    presents.append(present)
                    all_h.append(gather_2d(h, tf.expand_dims(final_id, axis=1))[:,0,:])
                presents = tf.stack(presents, axis=1)
                h = norm(h, 'ln_f')
                all_h.append(gather_2d(h, tf.expand_dims(final_id, axis=1))[:,0,:])
                target_mask = tf.sequence_mask(h_len, maxlen=tf.shape(h)[1], dtype=tf.float32)#如果是h_len-1则把sentence token给mask掉
                target_mask = tf.expand_dims(target_mask, 2)
                encode_out = tf.transpose(presents, perm=(0, 4, 2, 3, 1, 5))
                ori_enc_shape = tf.shape(encode_out)
                encode_out = tf.reshape(encode_out, shape=(tf.shape(presents)[0], tf.shape(presents)[4], -1))
                encode_out = tf.multiply(encode_out, target_mask)
                encode_out = tf.reshape(encode_out, shape=ori_enc_shape)
                encode_out = tf.transpose(encode_out, perm=(0, 4, 2, 3, 1, 5))
                encode_out.set_shape(past_shape(hparams=self.hparams, batch_size=None))
                return encode_out, all_h

    #for GPT-HA
    def encode(self, h, h_len, past=None, scope='encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                # Transformer
                presents = []
                pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
                assert len(pasts) == self.hparams.n_layer
                for layer, past_one in enumerate(pasts):
                    h, present = block(h, 'h%d' % layer, past=past_one, hparams=self.hparams)
                    presents.append(present)
                presents = tf.stack(presents, axis=1)
                h = norm(h, 'ln_f')
                final_id = h_len - 1
                h = gather_2d(h, tf.expand_dims(final_id, axis=1))
                target_mask = tf.sequence_mask(h_len-1, maxlen=tf.shape(h)[1], dtype=tf.float32)#h_len-1把sentence token给mask掉
                target_mask = tf.expand_dims(target_mask, 2)
                encode_out = tf.transpose(presents, perm=(0, 4, 2, 3, 1, 5))
                ori_enc_shape = tf.shape(encode_out)
                encode_out = tf.reshape(encode_out, shape=(tf.shape(presents)[0], tf.shape(presents)[4], -1))
                encode_out = tf.multiply(encode_out, target_mask)
                encode_out = tf.reshape(encode_out, shape=ori_enc_shape)
                encode_out = tf.transpose(encode_out, perm=(0, 4, 2, 3, 1, 5))
                encode_out.set_shape(past_shape(hparams=self.hparams, batch_size=None))
                return encode_out, h