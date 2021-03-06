"""
version 0.9 based on GPT-CAT 的multi_gpu_trainer
训练支持数据并行，梯度累计
将训练变量和推断变量分离存储
数据处理流程融合在该类中
version 1.0
取消训练变量和推断变量分离。这种做法并没有带来明显的好处，且不同的优化器会存储不同的训练变量，每一个都要专门写规则处理。
训练支持梯度累计
训练和推断支持数据并行
将数据处理与流程控制解耦，增加了新的datastream类来支持通用的数据处理pipeline，并在ModelFn中预留接口以支持特定任务的处理
只支持1个要优化的loss
支持输出多个需要观察的loss
只支持1个推断输出
支持加载预训练模型
version 1.1
支持多任务联合学习，多个loss，多个prediction
"""
import tensorflow as tf
from MyEstimator.ModelFn import ModelFn
import tensorflow.contrib.slim as slim
from tensorflow.python.util import nest
from tensorflow.python.training import optimizer
import numpy as np
from datetime import timedelta
import time
import os
from typing import List
from tensorflow.contrib import opt as optimizers_set
from MyEstimator.DataStream import DataStream
from typing import List, Dict, Any, Set, Tuple
from MyEstimator.utils import flatten_nested_list


class ModelEstimator:
    def __init__(self, device_id: List[int], model_fn: ModelFn):
        """
        self.train_op
        self.all_losses_ops_of_modelfn
        :param device_id:
        :param model_fn:
        """
        self.config = {}
        self.config['device_id'] = device_id
        self.config['optimizer'] = 'Adam'
        self.config['learning_rate'] = 1e-4
        self.config['weight_decay'] = 1e-10
        self.config['accum_var_scope'] = 'accum_var_scope'
        self.config['allow_soft_placement'] = True
        self.config['log_device_placement'] = False
        self.model_fn = model_fn
        self.graph = tf.Graph()
        # vars for data parallel:
        self.tower_grads = {}
        # vars for gradient accumlation:
        self.accum_vars = None
        self.accum_grad_ops = None
        self.zero_ops = None
        self.train_accum_grad_step = None
        # vars for save and restore
        self.vars_to_save = []
        tf.logging.set_verbosity(tf.logging.INFO)

    def compile_running_definitions_of_model_fn(self):
        # todo
        pass

    def set_device_id(self, device_id: List[int]):
        self.config['device_id'] = device_id

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
            grads = tf.concat(grads, 0)
            grad = tf.reduce_mean(grads, 0)
            grad_and_var = (grad, grad_and_vars[0][1])
            # [(grad0, var0),(grad1, var1),...]
            average_grads.append(grad_and_var)
        return average_grads

    def get_init_optimizer(self):
        if self.config['optimizer'] == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'AdaMax':
            return optimizers_set.AdaMaxOptimizer(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'NAdam':
            return optimizers_set.NadamOptimizer(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'AdamW':
            return optimizers_set.AdamWOptimizer(learning_rate=self.config['learning_rate'],
                                                 weight_decay=self.config['weight_decay'])
        else:  # SGD is default
            return tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])

    def build_data_parallel_training_graph(self, allow_gradient_accumulation: bool):
        with self.graph.as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.global_step = tf.get_variable(name='global_step', shape=(),
                                                   dtype=tf.int32, trainable=False,
                                                   initializer=tf.constant_initializer(1))
            self.opt = self.get_init_optimizer()
            for i in range(0, len(self.config['device_id'])):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
                    with tf.device('/gpu:%d' % self.config['device_id'][i]):
                        with tf.name_scope('parallel_%d' % i):
                            self.model_fn.create_placeholders(i)
                            self.model_fn.build_training_graph(group_id=i)
                            grads = nest.map_structure(lambda x: self.opt.compute_gradients(x),
                                                       self.model_fn.losses_groups[i])
                            for loss_name in grads.keys():
                                if loss_name in self.tower_grads:
                                    self.tower_grads[loss_name].append(grads[loss_name])
                                else:
                                    self.tower_grads[loss_name] = [grads[loss_name]]
            with tf.device('/gpu:0'):
                grads = {}
                self.train_ops = {}
                for loss_name in self.tower_grads.keys():
                    grads[loss_name] = self.average_gradients(self.tower_grads[loss_name])
                    self.train_ops[loss_name] = self.opt.apply_gradients(grads[loss_name],
                                                                         global_step=self.global_step)
                if allow_gradient_accumulation:
                    self.accum_grad_ops = {}
                    tvs = tf.trainable_variables()
                    with tf.variable_scope(self.config['accum_var_scope']):
                        self.accum_steps = tf.placeholder(tf.float32, [], name='accum_stpes')
                        self.accum_vars = [
                            tf.Variable(tf.zeros_like(tv.initialized_value(), name='Variable'), trainable=False)
                            for tv in tvs]
                        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
                        last_name = None
                        for loss_name in self.tower_grads.keys():
                            self.accum_grad_ops[loss_name] = [self.accum_vars[j].assign_add(gv[0]) for j, gv in
                                                              enumerate(grads[loss_name])]
                            last_name = loss_name
                        self.train_accum_grad_step = self.opt.apply_gradients(
                            [(self.accum_vars[j], gv[1]) for j, gv in enumerate(grads[last_name])],
                            global_step=self.global_step)
                # all_losses_ops_of_modelfn contains losses_to_optimize and losses_to_only_watch:
                self.all_losses_ops_of_modelfn = self.model_fn.get_items_group_by_name_from_by_id(
                    self.model_fn.losses_groups)
                self.all_losses_ops_of_modelfn.update(
                    self.model_fn.get_items_group_by_name_from_by_id(self.model_fn.losses_only_watch_groups))
                for loss_name in self.all_losses_ops_of_modelfn.keys():
                    self.all_losses_ops_of_modelfn[loss_name] = tf.reduce_mean(
                        tf.stack(self.all_losses_ops_of_modelfn[loss_name], axis=0))

    def build_data_parallel_inferring_graph(self):
        with self.graph.as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.global_step = tf.get_variable(name='global_step', shape=(),
                                                   dtype=tf.int32, trainable=False,
                                                   initializer=tf.constant_initializer(1))
            for i in range(0, len(self.config['device_id'])):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
                    with tf.device('/gpu:%d' % self.config['device_id'][i]):
                        with tf.name_scope('parallel_%d' % i):
                            self.model_fn.create_placeholders(i)
                            self.model_fn.build_inferring_graph(group_id=i)
            self.predictions_ops_of_model_fn = self.model_fn.get_items_group_by_name_from_by_id(
                self.model_fn.prediction_groups)
            for pred_name in self.predictions_ops_of_model_fn.keys():
                self.predictions_ops_of_model_fn[pred_name] = tf.stack(self.predictions_ops_of_model_fn[pred_name],
                                                                       axis=0)

    def load_checkpoint(self, path, logging) -> Dict[str, Any]:
        """
        load a checkpoint
        :param path: checkpoint path or its parent path
        :return:
        """
        ckpt = tf.train.latest_checkpoint(path)
        self.print_and_logging("Loading %s" % ckpt, logging=logging)
        var_list = tf.train.list_variables(ckpt)
        values = {}
        reader = tf.train.load_checkpoint(ckpt)
        total_size = 0
        for (name, shape) in var_list:
            tensor = reader.get_tensor(name)
            values[name] = tensor
            self.print_and_logging(content="pre-trained variable: %s\tshape    %s" % (name.ljust(80),
                                                                                      str(tensor.shape).ljust(20)),
                                   logging=logging)
            v_size = np.prod(np.array(tensor.shape)).tolist()
            total_size += v_size
        self.print_and_logging(content="Total pre-trained variables size: %d" % total_size, logging=logging)
        return values

    def create_session_init_and_print_all_vars(self, max_to_save, pretrained_ckpt, logging=False):
        # Print parameters
        with self.graph.as_default():
            all_trainable_weights = {v.name: v for v in tf.trainable_variables()}
            total_size = 0
            for v_name in sorted(list(all_trainable_weights)):
                v = all_trainable_weights[v_name]
                self.print_and_logging(content="trainable variable: %s\tshape    %s" % (v.name[:-2].ljust(80),
                                                                                        str(v.shape).ljust(20)),
                                       logging=logging)
                v_size = np.prod(np.array(v.shape.as_list())).tolist()
                total_size += v_size
            self.print_and_logging("Total trainable variables size: %d" % total_size, logging=logging)
            all_var_list = slim.get_variables_to_restore()
            total_size = 0
            for v in all_var_list:
                if v.name.startswith(self.config['accum_var_scope']):
                    self.print_and_logging(content="accum variable: %s\tshape    %s" % (v.name[:-2].ljust(80),
                                                                                        str(v.shape).ljust(20)),
                                           logging=logging)
                    v_size = np.prod(np.array(v.shape.as_list())).tolist()
                    total_size += v_size
                else:
                    self.vars_to_save.append(v)
            self.print_and_logging("Total accum variables size: %d" % total_size, logging=logging)
            total_size = 0
            for v in self.vars_to_save:
                self.print_and_logging("stored variable: %s\tshape    %s" % (v.name[:-2].ljust(80),
                                                                             str(v.shape).ljust(20)), logging=logging)
                v_size = np.prod(np.array(v.shape.as_list())).tolist()
                total_size += v_size
            self.print_and_logging("Total stored variables size: %d" % total_size, logging=logging)
            if len(self.vars_to_save) > 0:
                self.saver = tf.train.Saver(self.vars_to_save, max_to_keep=max_to_save)
            config = tf.ConfigProto(allow_soft_placement=self.config['allow_soft_placement'],
                                    log_device_placement=self.config['log_device_placement'])
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)
            # raw_init
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # load pre_trained parameters:
            loaded_pretrained_vars = {}
            if pretrained_ckpt is not None:
                mapping_dict = self.model_fn.vars_mapping_for_loading_transfer_param(self.vars_to_save)
                pretrained_data = self.load_checkpoint(pretrained_ckpt, logging=logging)
                restore_ops = []
                for v in self.vars_to_save:
                    if mapping_dict[v.name] in pretrained_data:
                        op = tf.assign(v, pretrained_data[mapping_dict[v.name]])
                        restore_ops.append(op)
                        loaded_pretrained_vars[mapping_dict[v.name]] = v
                for name in pretrained_data.keys():
                    if name not in loaded_pretrained_vars:
                        self.print_and_logging('NOTICE: %s in Pre-trained checkpoint is not loaded.' % name,
                                               logging=logging)
                sess.run(restore_ops)
                self.print_and_logging('Loaded %d variables from pre-trained checkpoint.' % len(restore_ops),
                                       logging=logging)
            return sess

    def restore_model(self, sess, ckpt_dir, logging):
        with self.graph.as_default():
            ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if ckpt is not None:
                self.saver.restore(sess, ckpt)
                self.print_and_logging('restored params from %s' % ckpt, logging=logging)

    def save_model(self, sess, ckpt_dir, step):
        with self.graph.as_default():
            if ckpt_dir is not None and len(self.vars_to_save) > 0:
                self.saver.save(sess, os.path.join(ckpt_dir, self.model_fn.__class__.__name__), global_step=step)

    def train_one_batch(self, sess, train_data_stream: DataStream,
                        batch_size: int, mini_batch: int, train_steps_for_this_batch: List[List[str]],
                        run_options=None) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        train one batch, with multi gpus and grad accum
        train steps are in train_steps_for_this_batch
        :param sess: tensorflow session
        :param train_data_stream: confirms that feed batch_size data to the model
        :param batch_size: data num for this training step
        :param mini_batch: num of data which is feeded to one gpu every time.
        :param train_steps_for_this_batch
        :param run_options: tensorflow running options
        :return:
        """
        # batchsize=minibatch*devicenum*k
        device_num = len(self.config['device_id'])
        accum_steps = batch_size // (mini_batch * device_num)
        all_losses_in_this_batch = set([x for x in flatten_nested_list(train_steps_for_this_batch)])
        all_needed_pls = {}
        for loss_name in all_losses_in_this_batch:
            all_needed_pls[loss_name] = self.model_fn.placeholder_requirement_for_losses[loss_name]
        all_feed_data = []
        for i in range(0, accum_steps):
            tmp = {}
            for j in range(0, device_num):
                tmp.update(train_data_stream.get_feed_dict(self.model_fn.placeholder_groups[j], size=mini_batch,
                                                           op_name_to_run_and_target_data_name=all_needed_pls,
                                                           modelfn_post_process=self.model_fn.feed_dict_post_process))
            all_feed_data.append(tmp)
        ret_losses = {}
        losses_occurance = {}
        for loss_name in all_losses_in_this_batch:
            ret_losses[loss_name] = 0.0
            losses_occurance[loss_name] = 0
        # some var refs:
        all_losses_ops_of_modelfn = self.all_losses_ops_of_modelfn
        model_fn = self.model_fn
        train_ops = self.train_ops
        accum_grad_ops = self.accum_grad_ops
        graph = self.graph

        # train one step:
        def train_one_step(one_step_losses: Set[str]):
            can_be_optimized_train_directly = {}
            can_be_optimized_grad_accum = {}
            losses_to_watch = {}
            placeholder_requirement = {}
            for loss_name in one_step_losses:
                losses_to_watch[loss_name] = all_losses_ops_of_modelfn[loss_name]
                placeholder_requirement[loss_name] = model_fn.placeholder_requirement_for_losses[loss_name]
                if loss_name in train_ops:
                    if accum_grad_ops is None:
                        can_be_optimized_train_directly[loss_name] = train_ops[loss_name]
                    else:
                        can_be_optimized_grad_accum[loss_name] = accum_grad_ops[loss_name]
            needed_pls_name_list = [placeholder_requirement[k] for k in placeholder_requirement.keys()]
            needed_pls_name_list = set(flatten_nested_list(needed_pls_name_list))
            # gradient accum and update
            with graph.as_default():
                feed_dict = {}
                if accum_steps == 1:
                    for j in range(0, device_num):
                        for pls_name in needed_pls_name_list:
                            pls = model_fn.placeholder_groups[j][pls_name]
                            feed_dict[pls] = all_feed_data[0][pls]
                    result = sess.run([can_be_optimized_train_directly, losses_to_watch],
                                      feed_dict=feed_dict, options=run_options)
                    losses_values = result[-1]
                    for loss_name in losses_values.keys():
                        ret_losses[loss_name] += losses_values[loss_name]
                        losses_occurance[loss_name] += 1
                else:
                    sess.run([self.zero_ops])
                    for i in range(0, accum_steps):
                        for j in range(0, device_num):
                            for pls_name in needed_pls_name_list:
                                pls = model_fn.placeholder_groups[j][pls_name]
                                feed_dict[pls] = all_feed_data[i][pls]
                        result = sess.run([can_be_optimized_grad_accum, losses_to_watch],
                                          feed_dict=feed_dict, options=run_options)
                        losses_values = result[-1]
                        for loss_name in losses_values.keys():
                            ret_losses[loss_name] += losses_values[loss_name]
                            losses_occurance[loss_name] += 1
                    sess.run(self.train_accum_grad_step, feed_dict={self.accum_steps: accum_steps})

        for one_training_step in train_steps_for_this_batch:
            train_one_step(set(one_training_step))
        return ret_losses, losses_occurance

    def eval_for_one_dataset(self, sess: tf.Session, eval_data_stream: DataStream,
                             mini_batch: int, run_options=None) -> Dict[str, float]:
        """
        eval on one dataset
        fetch all losses in model_fn.eval_steps
        :param sess:
        :param eval_data_stream:
        :param mini_batch:
        :param run_options:
        :return: {'loss_1':0.05,...}
        """
        model_fn = self.model_fn
        graph = self.graph
        device_num = len(self.config['device_id'])
        data_num = eval_data_stream.dataset_size
        assert data_num >= device_num, 'do not suppose this case: data_num<device_num'
        # to allocate data for each gpu each step
        yu = data_num % (mini_batch * device_num)
        normal_steps = data_num // (mini_batch * device_num)
        sp_steps_data_num = []
        if yu != 0:
            if yu < device_num:
                normal_steps -= 1
                for i in range(0, yu):
                    sp_steps_data_num.append(mini_batch + 1)
                for i in range(yu, device_num):
                    sp_steps_data_num.append(mini_batch)
            else:
                sp_steps_data_num = [yu // device_num] * device_num
                for i in range(0, yu % device_num):
                    sp_steps_data_num[i] += 1
        all_losses_ops_of_modelfn = self.all_losses_ops_of_modelfn

        # eval one step:
        def eval_one_step(losses_to_fetch: Set[str]) -> Dict[str, float]:
            fetch_dict = {}
            for loss_name in losses_to_fetch:
                fetch_dict[loss_name] = all_losses_ops_of_modelfn[loss_name]
            pls_for_op = {}
            for op_name in losses_to_fetch:
                pls_for_op[op_name] = model_fn.placeholder_requirement_for_losses[op_name]
            with graph.as_default():
                feed_dict = {}
                # calculate losses
                ret_losses = {}
                for loss_name in losses_to_fetch:
                    ret_losses[loss_name] = 0.0
                for i in range(0, normal_steps):
                    feed_dict.clear()
                    for j in range(0, device_num):
                        feed_dict.update(eval_data_stream. \
                                         get_feed_dict(model_fn.placeholder_groups[j],
                                                       size=mini_batch,
                                                       op_name_to_run_and_target_data_name=pls_for_op,
                                                       modelfn_post_process=model_fn.feed_dict_post_process))
                    result = sess.run(fetch_dict, feed_dict=feed_dict, options=run_options)
                    for loss_name in losses_to_fetch:
                        ret_losses[loss_name] += result[loss_name] * device_num * mini_batch
                feed_dict.clear()
                for j in range(0, device_num):
                    feed_dict.update(eval_data_stream. \
                                     get_feed_dict(model_fn.placeholder_groups[j],
                                                   size=sp_steps_data_num[j],
                                                   op_name_to_run_and_target_data_name=pls_for_op,
                                                   modelfn_post_process=model_fn.feed_dict_post_process))
                result = sess.run(fetch_dict, feed_dict=feed_dict, options=run_options)
                for loss_name in losses_to_fetch:
                    ret_losses[loss_name] += result[loss_name] * sum(sp_steps_data_num)
                for loss_name in losses_to_fetch:
                    ret_losses[loss_name] /= data_num
            assert eval_data_stream.dataset_size == eval_data_stream.low, 'data allocation may be wrong'
            eval_data_stream.reset_feeding_status()
            return ret_losses

        ret = {}
        for losses_in_one_eval_step in self.model_fn.eval_steps:
            ret.update(eval_one_step(losses_to_fetch=set(losses_in_one_eval_step)))
        return ret

    def predict_for_one_dataset(self, sess: tf.Session, test_data_stream: DataStream,
                                mini_batch: int, run_options=None):
        """
        predict for one dataset
        fetch all predictions in model_fn.predicting_steps
        :param sess:
        :param test_data_stream:
        :param mini_batch:
        :param run_options:
        :return:
        """
        device_num = len(self.config['device_id'])
        data_num = test_data_stream.dataset_size
        assert data_num >= device_num, 'do not suppose this case: data_num<device_num'
        # to allocate data for each gpu each step
        yu = data_num % (mini_batch * device_num)
        normal_steps = data_num // (mini_batch * device_num)
        sp_steps_data_num = []
        if yu != 0:
            if yu < device_num:
                normal_steps -= 1
                for i in range(0, yu):
                    sp_steps_data_num.append(mini_batch + 1)
                for i in range(yu, device_num):
                    sp_steps_data_num.append(mini_batch)
            else:
                sp_steps_data_num = [yu // device_num] * device_num
                for i in range(0, yu % device_num):
                    sp_steps_data_num[i] += 1
        graph = self.graph
        model_fn = self.model_fn
        predictions_ops_of_model_fn = self.predictions_ops_of_model_fn
        return_data = {}

        def predict_one_step(predictions_of_one_step: Set[str]):
            pls_for_op = {}
            fetch_dict = {}
            for op_name in predictions_of_one_step:
                pls_for_op[op_name] = model_fn.placeholder_requirement_for_predictions[op_name]
                fetch_dict[op_name] = predictions_ops_of_model_fn[op_name]
            with graph.as_default():
                feed_dict = {}
                # calculate predictions
                ret_predictions = None
                for i in range(0, normal_steps):
                    feed_dict.clear()
                    for j in range(0, device_num):
                        feed_dict.update(
                            test_data_stream.get_feed_dict(model_fn.placeholder_groups[j], size=mini_batch,
                                                           op_name_to_run_and_target_data_name=pls_for_op,
                                                           modelfn_post_process=model_fn.feed_dict_post_process))
                    result = sess.run(fetch_dict, feed_dict=feed_dict, options=run_options)
                    ret_predictions = model_fn.merge_batch_prediction_result(result, ret_predictions)
                feed_dict.clear()
                for j in range(0, device_num):
                    feed_dict.update(
                        test_data_stream.get_feed_dict(model_fn.placeholder_groups[j], size=sp_steps_data_num[j],
                                                       op_name_to_run_and_target_data_name=pls_for_op,
                                                       modelfn_post_process=model_fn.feed_dict_post_process))
                result = sess.run(fetch_dict, feed_dict=feed_dict, options=run_options)
                ret_predictions = model_fn.merge_batch_prediction_result(result, ret_predictions)
            assert test_data_stream.dataset_size == test_data_stream.low, 'data allocation may be wrong'
            test_data_stream.reset_feeding_status()
            return ret_predictions

        for predictions_of_one_step in self.model_fn.predicting_steps:
            return_data.update(predict_one_step(set(predictions_of_one_step)))
        return return_data

    def training(self, train_stream: DataStream, dev_stream: DataStream, ckpt_dir,
                 learning_rate=1e-4, batch_size=64, mini_batch=16, total_steps=100000,
                 eval_per_n_steps=1, max_to_save=3,
                 early_stop_steps=6000, pretrained_ckpt=None):
        device_num = len(self.config['device_id'])
        assert batch_size % device_num == 0
        assert batch_size >= mini_batch * device_num and batch_size % (mini_batch * device_num) == 0
        assert type(train_stream.text_index_encoder) == type(
            dev_stream.text_index_encoder), 'bpe tool must be same for train and dev'
        # set vocab size for modelfn
        self.model_fn.set_vocab_size(dev_stream.text_index_encoder.get_vocab_size())
        # set learning rate
        self.config['learning_rate'] = learning_rate
        # bulid graph and init
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.log_file = open(ckpt_dir + 'log-%s.txt' % (get_time_stamp()), 'w', encoding='utf-8')
        if batch_size == mini_batch * len(self.config['device_id']):
            self.build_data_parallel_training_graph(allow_gradient_accumulation=False)
        else:
            self.build_data_parallel_training_graph(allow_gradient_accumulation=True)
        self.model_fn.check_after_build_graph()
        # continue train or not?
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            pretrained_ckpt = None
        # create session
        sess = self.create_session_init_and_print_all_vars(max_to_save, pretrained_ckpt, logging=True)
        # restore model
        self.restore_model(sess, ckpt_dir=ckpt_dir, logging=True)
        # vars for training loop
        step = sess.run(self.global_step)
        last_improvement_step = step - 1
        self.print_and_logging('LPT:%d' % last_improvement_step)
        best_loss = None
        saved_steps = []
        self.print_and_logging('start training...')
        self.graph.finalize()
        start_time = time.time()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        while step < total_steps:
            train_loss = {}
            train_loss_occur = {}
            for batch_train_steps in self.model_fn.training_steps:
                tl, tlc = self.train_one_batch(sess, train_stream, batch_size=batch_size,
                                               run_options=run_options,
                                               mini_batch=mini_batch,
                                               train_steps_for_this_batch=batch_train_steps)
                for name in tl.keys():
                    if name not in train_loss:
                        assert name not in train_loss_occur
                        train_loss[name] = tl[name]
                        train_loss_occur[name] = tlc[name]
                    else:
                        train_loss[name] += tl[name]
                        train_loss_occur[name] += tlc[name]
            for name in train_loss.keys():
                train_loss[name] /= train_loss_occur[name]
            ###eval:
            if step % eval_per_n_steps == 0:
                eval_loss = self.eval_for_one_dataset(sess, dev_stream, mini_batch=mini_batch, run_options=run_options)
                time_dif = get_time_dif(start_time)
                if self.new_losses_are_better(new_losses=eval_loss, old_losses=best_loss):
                    best_loss = eval_loss
                    last_improvement_step = step
                    self.print_and_logging('save step %d' % last_improvement_step)
                    self.save_model(sess, ckpt_dir=ckpt_dir, step=step)
                    saved_steps.append(last_improvement_step)
                    self.print_and_logging("%s: step %d: *\ntrain loss: %s *\neval loss: %s *" %
                                           (time_dif, step, self.get_obvious_loss_report(train_loss),
                                            self.get_obvious_loss_report(eval_loss)))
                    if len(saved_steps) > max_to_save:
                        saved_steps = saved_steps[1:]
                else:
                    self.print_and_logging("%s: step %d:\ntrain loss: %s\neval loss: %s" %
                                           (time_dif, step, self.get_obvious_loss_report(train_loss),
                                            self.get_obvious_loss_report(eval_loss)))
                    if step - last_improvement_step > early_stop_steps:
                        self.print_and_logging("early stopping...")
                        break
            ###
            step += 1
        print('all work has finished')

    def inferring(self, data_stream: DataStream, ckpt_dir, mini_batch=16, logging=False):
        assert os.path.exists(ckpt_dir)
        assert tf.train.latest_checkpoint(ckpt_dir) is not None
        if logging:
            self.log_file = open(ckpt_dir + 'infer_log-%s.txt' % (get_time_stamp()), 'w', encoding='utf-8')
        # set vocab size for modelfn
        self.model_fn.set_vocab_size(data_stream.text_index_encoder.get_vocab_size())
        self.build_data_parallel_inferring_graph()
        self.model_fn.check_after_build_graph()
        sess = self.create_session_init_and_print_all_vars(max_to_save=1, pretrained_ckpt=None, logging=logging)
        self.restore_model(sess=sess, ckpt_dir=ckpt_dir, logging=logging)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return self.predict_for_one_dataset(sess=sess, test_data_stream=data_stream, mini_batch=mini_batch,
                                            run_options=run_options)

    def print_and_logging(self, content: str, logging=True):
        tf.logging.info(content)
        if logging:
            self.log_file.write(content.strip() + '\n')
            self.log_file.flush()

    def get_obvious_loss_report(self, losses: Dict[str, float]):
        return str(losses)

    def new_losses_are_better(self, new_losses, old_losses):
        if old_losses is None:
            return True
        return self.model_fn.new_losses_are_better(new_losses=new_losses, old_losses=old_losses)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_time_stamp():
    now = int(round(time.time() * 1000))
    return time.strftime('%Y%m%d-%H%M%S', time.localtime(now / 1000))
