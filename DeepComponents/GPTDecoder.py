import tensorflow as tf
from DeepComponents.TransformerBlock import block, positions_for,norm, shape_list
from DeepComponents.common_ops import gather_2d, tile_to_beam_size, merge_first_two_dims

class Decoder():
    def __init__(self,scope,hparams):
        self.scope = scope
        self.hparams = hparams
        with tf.variable_scope(scope):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                self.wpe=tf.get_variable('wpe', [self.hparams.n_ctx, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
                self.wte = tf.get_variable('wte', [self.hparams.n_vocab, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.02))
                self.attn_w = tf.get_variable(shape=(self.hparams.n_embd, self.hparams.n_embd), name='sen_attn_w')


    def decode_all(self,tokens,past_list,enc_h_list):
        """for multiple sources, like GPT-HA, if len(past_list)==1, it is a simple GPTEncoder-Decoder model"""
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
                results = {}
                if type(past_list)!=list:
                    past_list=[past_list]
                batch, sequence = shape_list(tokens)
                #past_length = 0
                all_past_length=[0 if past_list[0] is None else tf.shape(past_list[0])[-2]]
                past_length = tf.reduce_max(tf.stack(all_past_length,axis=0),axis=0)
                h = tf.gather(self.wte, tokens) + tf.gather(self.wpe, positions_for(tokens, past_length))
                values_present = {}
                for i in range(0, self.hparams.n_layer):
                    querys = h
                    values_h = []
                    for j in range(0, len(past_list)):
                        past = past_list[j]
                        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
                        assert len(pasts) == self.hparams.n_layer
                        h, present = block(querys, 'h%d' % i, past=pasts[i], hparams=self.hparams)
                        values_h.append(h)
                        if j in values_present:
                            values_present[j].append(present)
                        else:
                            values_present[j]=[present]
                    enc_h_all = tf.concat(enc_h_list, axis=1)
                    attn_score = tf.tensordot(querys, self.attn_w, axes=(2, 0))
                    attn_score = tf.matmul(attn_score, tf.transpose(enc_h_all, perm=(0, 2, 1)))  # batch*seq*context_num
                    attn_score = tf.nn.softmax(attn_score,axis=2)
                    val_h_cat = tf.stack(values_h, axis=2)
                    val_h_cat = tf.expand_dims(attn_score, axis=3) * val_h_cat
                    val_h_cat = tf.reduce_sum(val_h_cat, axis=2)
                    h = val_h_cat
                for j in range(0,len(past_list)):
                    values_present[j]=tf.stack(values_present[j],axis=1)
                    past_list[j]=tf.concat([past_list[j],values_present[j]],axis=-2)
                h = norm(h, 'ln_f')
                # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
                logits = tf.matmul(h_flat, self.wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
                results['logits'] = logits
                return results


    def sef_var_for_beam_search(self,enc_0_len,enc_h_list,beam_size):
        self.enc_0_len=enc_0_len
        self.enc_h_list=enc_h_list
        self.enc_h_all = tf.concat(self.enc_h_list, axis=1)
        self.enc_h_all=merge_first_two_dims(tile_to_beam_size(self.enc_h_all,beam_size=beam_size))


    def decode_one_step(self,hparams:"no use, only for consistency of api", input_token, past_dec:list):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                all_past_length = [0 if past_dec[j] is None else tf.shape(past_dec[j])[-2] for j in range(0,len(past_dec))]
                past_length=tf.reduce_max(tf.stack(all_past_length,axis=0),axis=0)
                h = tf.gather(self.wte, input_token) + tf.gather(self.wpe, positions_for(input_token, past_length))
                results = {}
                batch, sequence = shape_list(input_token)
                values_present = {}
                for i in range(0, self.hparams.n_layer):
                    querys = h
                    values_h = []
                    for j in range(0, len(past_dec)):
                        dec_pasts = tf.unstack(past_dec[j], axis=1) if past_dec[j] is not None else [None] * self.hparams.n_layer  #
                        h, present = block(querys, 'h%d' % i,
                                           past=dec_pasts[i],
                                           hparams=self.hparams)
                        values_h.append(h)
                        if j in values_present:
                            values_present[j].append(present)
                        else:
                            values_present[j]=[present]
                    attn_score = tf.tensordot(querys, self.attn_w, axes=(2, 0))
                    attn_score = tf.matmul(attn_score, tf.transpose(self.enc_h_all, perm=(0, 2, 1)))  # batch*seq*context_num
                    attn_score = tf.nn.softmax(attn_score, axis=2)
                    val_h_cat = tf.stack(values_h, axis=2)
                    val_h_cat = tf.expand_dims(attn_score, axis=3) * val_h_cat
                    val_h_cat = tf.reduce_sum(val_h_cat, axis=2)
                    h = val_h_cat
                for j in range(0,len(past_dec)):
                    values_present[j]=tf.stack(values_present[j],axis=1)
                    past_dec[j]=tf.concat([past_dec[j],values_present[j]],axis=-2)
                h = norm(h, 'ln_f')
                # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
                logits = tf.matmul(h_flat, self.wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
                results['logits'] = logits
                results['presents']= past_dec
                return results