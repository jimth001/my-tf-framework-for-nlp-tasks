import tensorflow as tf
from MyEstimator.ModelFn import ModelFn, PlaceholderMetaData, PlaceholderType
from DeepComponents.GPT import model
from tensorflow.contrib.seq2seq import sequence_loss
from DeepComponents.TransformerBlock import default_hparams
from typing import List,Dict,Any
from DeepComponents.BeamSearch import create_inference_graph
import numpy as np


class GPTModel(ModelFn):
    def __init__(self):
        super(GPTModel, self).__init__()
        self.init_config()

    def init_config(self):
        meta_data_of_placeholders = {
            'input': PlaceholderMetaData(PlaceholderType.Text, shape=[None, None], dtype=tf.int32),
            'input_len': PlaceholderMetaData(PlaceholderType.TextLength, Ref='input', shape=[None, ], dtype=tf.int32),
            'target': PlaceholderMetaData(PlaceholderType.Text, shape=[None, None], dtype=tf.int32),
            'target_len': PlaceholderMetaData(PlaceholderType.TextLength, Ref='target', shape=[None, ], dtype=tf.int32),
            'target_mask': PlaceholderMetaData(PlaceholderType.TextTargMask, Ref='target', shape=[None, None],
                                               dtype=tf.float32),
            'batch_size': PlaceholderMetaData(PlaceholderType.BatchSize, Ref=None, shape=(), dtype=tf.int32)
        }
        """about 'target_mask':
        target_mask let model can only learn to predict a subsquence,
        if a token is masked in target_mask, it will not be considered in the LM loss.
        Note that in GPTModel, different sentence will be concat directly, 
        so if you want do a seq2seq prediction, you will need target_mask to mask input_sentence.
        """
        self.config['placeholders'] = meta_data_of_placeholders
        self.config['hparams'] = default_hparams()
        self.config['only_for_pretraining'] = False

    def set_vocab_size(self,vocab_size:int):
        self.config['hparams'].set_hparam(name='n_vocab',value=vocab_size)

    def build_training_graph(self, group_id):
        placeholders = self.placeholder_groups[group_id]
        result = model(hparams=self.config['hparams'], X=placeholders['input'],
                       scope=self.__class__.__name__, reuse=tf.AUTO_REUSE)
        logits = result['logits']
        if self.config['only_for_pretraining']:
            target_mask = tf.sequence_mask(placeholders['target_len'], dtype=tf.float32)
        else:  # maybe for seq2seq, when user defined target_mask is needed.
            target_mask = placeholders['target_mask']
        cost = sequence_loss(logits=logits, targets=placeholders['target'],
                             weights=target_mask)
        self.losses_groups[group_id] = {'gpt_lm_loss': cost}
        return cost

    def build_inferring_graph(self, group_id):
        with tf.variable_scope(self.__class__.__name__) as m_scope:
            placeholders=self.placeholder_groups[group_id]

            def step_fn(hparam, tokens, past=None, scope=m_scope):
                output=model(hparams=hparam,X=tokens,past=past,scope=scope,reuse=tf.AUTO_REUSE)
                present = output['presents']
                output['presents'] = tf.concat([past, present], axis=-2)
                return output
            context_state=model(hparams=self.config['hparams'],X=placeholders['input'],past=None,scope=m_scope,reuse=tf.AUTO_REUSE)
            """
            training:
             input:      a b c     <eos> d e f
             target:     b c <eos> d     e f <eos>
             target mask:0 0 0     1     1 1 1
             inferring:
             input:      a b c     <eos>  #(so context_state should drop the <eos> result.)
             init_seq:   <eos>
            """
            context_state = context_state['presents'][:, :, :, :, :-1, :]
            init_seq = tf.fill(dims=(placeholders['batch_size'], 1), value=self.config['eos_id'])
            seqs,scores=create_inference_graph(init_seqs=init_seq,
                                               state=context_state,
                                               step_fn=step_fn,
                                               hparams=self.config['hparams'],
                                               decode_length=self.config['decode_length'],
                                               batch_size=placeholders['batch_size'],
                                               beam_size=self.config['beam_size'],
                                               decode_alpha=self.config['decode_alpha'],
                                               eos_id=self.config['eos_id'], ensemble=False,
                                               concat_state_dim=None, scopes_for_ensemble=None)
            return seqs

    #task specified
    def process_origin_data_for_placeholders(self, data: Dict[str, List[Any]], for_loss_n:str=None) -> Dict[str, List[Any]]:
        new_data={}
        if 'target' in data:
            new_data['input'] = [i + t[:-1] for i, t in zip(data['input'], data['target'])]
            new_data['target']=[i[1:]+t for i,t in zip(data['input'],data['target'])]
        else:
            new_data['input'] = data['input']
        for name in self.config['placeholders'].keys():
            pl_meta:PlaceholderMetaData=self.config['placeholders'][name]
            if pl_meta.Ref in data:
                if pl_meta.type == PlaceholderType.TextLength:
                    ref_data: List = new_data[pl_meta.Ref]
                    new_data[name] = [len(s) for s in ref_data]
                elif pl_meta.type == PlaceholderType.TextTargMask:
                    ref_data_old: List[List] = data[pl_meta.Ref]
                    ref_data_new: List[List] = new_data[pl_meta.Ref]
                    new_data[name] = [[0.] * (len(n) - len(o)) + [1.] * len(o) for o, n in
                                      zip(ref_data_old, ref_data_new)]
        return new_data

    def new_losses_are_better(self,new_losses:List,old_losses:List, losses_name:List):
        id=losses_name.index('gpt_lm_loss')
        return new_losses[id]<old_losses[id]

    def vars_mapping_for_loading_transfer_param(self, vars_to_store:List[tf.Variable]) -> Dict[str, str]:
        """
        model specific, for loading gpt2 pretrained checkpoints
        you can overload it to adapt to other checkpoints
        :param vars_to_store:
        :return:
        """
        d={}
        for v in vars_to_store:
            if v.name=='global_step':
                d[v.name]='**None**//'#do not load global_step from pretrained checkpoint
            d[v.name]='model'+v.name[len(self.__class__.__name__):-2]
        return d

    def merge_batch_prediction_result(self, new_batch_result: Dict[str, np.array],
                                      previous_result: Dict[str, List] or None):
        if previous_result is None:
            ret = {}
            for key in new_batch_result.keys():
                ret[key] = new_batch_result[key].tolist()
            return ret
        else:
            for key in previous_result.keys():
                previous_result[key] += new_batch_result[key].tolist()
            return previous_result

    def parse_out_idx(self, index_list: List[int], eos_id: int):
        r = []
        index_list = index_list[1:]
        for x in index_list:
            if x != eos_id:
                r.append(x)
            else:
                break
        return r
