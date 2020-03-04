from abc import ABCMeta, abstractmethod
from tensorflow.python.util import nest
import tensorflow as tf
from collections import OrderedDict
from enum import Enum
import json
from typing import List, Dict, Any, Collection, NoReturn, Set, Union


class PlaceholderType(Enum):
    Text = 1
    TargetInt = 2
    TargetFloat = 3
    TextLength = 4
    TextAttnMask = 5
    TextTargMask = 6
    BatchSize = 7
    SpecialBatchInformation = 8


class PlaceholderMetaData():
    def __init__(self, type: PlaceholderType, dtype, shape, from_file_name=None, Ref=None):
        """
        :param type: source input type for this placeholder
        :param dtype: dtype for tf.placeholder
        :param shape: shape for tf.placeholder
        :param from_file_name: source data file's name for this placeholder
                There are two ways to load data:
                1.Read data for each placeholder from each from_file_name.
                2.Read all data from one tsv file. The first line of the tsv must be the names of corresponding placeholders.
        :param Ref: if Ref is None, this placeholder will not depend on other placeholder, such as Text.
                     For TextLength placeholder which depend on a Text placeholder, its Ref value should be
                     the name of the Text placeholder.
        """
        self.type = type
        self.Ref = Ref
        self.shape = shape
        self.from_file_name = from_file_name
        self.dtype = dtype


class ModelFn:
    def __init__(self):
        """
            A model could have n placeholders, m losses to optimize,
            k losses to only watch, t predictions to output.
            Every placeholder, loss, prediction should have a name.
            Each loss requires x placeholders, which can be represented as {loss_name:[placeholder1,...]}.
            Each prediction requires x placeholders, which can be represented as {loss_name:[placeholder1,...]}.

            Different losses, predictions, optimize_ops may have conflicts so sometimes you can not running once to fetch all of them.
            Although this is almost not to happen, we also provide a strategy to deal with it.
            In eval stage, we apply an list named eval_steps to define how to eval all losses in n steps.
                n=len(eval_steps)
                eval_steps=[step1,step2,...stepn]
                step_i is a list of losses
                You can put conflict losses into different steps.
                Notice that use more than 1 eval steps will cause additional computational overhead,
                so you should use it only when necessary.
            In training stage, similarly but differently, we also provide a "training_steps" list, where
                training_steps=[batch_1_step,batch_2_step,...]
                batch_1_step=[step1,step2,...]
                step1=[loss1,...] ...
                In training stage:
                    for one_batch_steps in training_steps:
                        produce one batch data
                        for losses in one_batch_steps:
                            split losses into can_be_optimized and only_watch
                            then train(can_be_optimized,data) and fetch(only_watch,data)
            In prediction stage, there is a list named "predicting_steps", similar to "eval_steps".

            In training stage, we create a train_op for each optimized loss.
            In training steps, if a loss is in self.losses_groups, ModelWrapper will run the corresponding train_op
            and fetch the loss value, otherwise ModelWrapper will only fetch the loss value to display.
        """
        # initialized when create placeholders
        self.placeholder_groups: Dict[int, Dict[str, tf.placeholder]] = OrderedDict()
        # initialized when building training graph
        self.losses_groups: Dict[int, Dict[str, any]] = OrderedDict()
        # initialized when building training graph
        self.losses_only_watch_groups: Dict[int, Dict[str, any]] = OrderedDict()
        # initialized when building inference graph
        # NOTICE that all prediction tensor should be batch first.
        self.prediction_groups: Dict[int, Dict[str, any]] = OrderedDict()
        # initialized when the object is constructed. self.init_check() can help to check some errors.
        self.placeholder_requirement_for_losses: Dict[str, List[str]] = {}
        self.placeholder_requirement_for_predictions: Dict[str, List[str]] = {}
        self.placeholders_meta_data: Dict[str, PlaceholderMetaData] = {}
        self.training_steps: List[List[List[str]]] = []  # [batch1=[ step1=loss1,step2=loss3,... ] ]
        self.eval_steps: List[List[str]] = []  # [first_step=[loss1,loss3],second_step=[loss2]]
        self.predicting_steps: List[List[str]] = []  # [first_step=[pred1,pred2],second_step=[pred3]]
        self.config: Dict[str, Any] = {}
        # self.batch_train_steps_pointer = 0

    def check_after_init(self):
        # check if there are obvious errors after init.
        # check type and length
        def check_type_and_len(var: Collection, var_type, least_elements: int):
            assert var is not None, "var can not be None"
            assert type(var) == var_type, "Wrong type, should be %s" % (str(var_type))
            assert len(var) > least_elements, "%s must have more than %d elements" % (str(var_type), least_elements)

        check_type_and_len(self.placeholders_meta_data, dict, 0)
        check_type_and_len(self.placeholder_requirement_for_losses, dict, 0)
        for k, v in self.placeholder_requirement_for_losses.items():
            check_type_and_len(v, list, 0)
        check_type_and_len(self.placeholder_requirement_for_predictions, dict, -1)
        for k, v in self.placeholder_requirement_for_predictions.items():
            check_type_and_len(v, list, 0)
        check_type_and_len(self.config, dict, -1)
        check_type_and_len(self.training_steps, list, 0)
        for batch_training_steps in self.training_steps:
            check_type_and_len(batch_training_steps, list, 0)
            for one_step_losses in batch_training_steps:
                check_type_and_len(one_step_losses, list, 0)
        # all required placeholders exist:
        required_placeholders_set = set()
        for loss_name in self.placeholder_requirement_for_losses.keys():
            required_placeholders_set = required_placeholders_set | set(
                self.placeholder_requirement_for_losses[loss_name])
        for pred_name in self.placeholder_requirement_for_predictions.keys():
            required_placeholders_set = required_placeholders_set | set(
                self.placeholder_requirement_for_predictions[pred_name])
        exist_placeholders_1 = set([name for name in self.placeholders_meta_data.keys()])
        assert len(required_placeholders_set) == len(exist_placeholders_1), "all required placeholders should exist"
        assert len(required_placeholders_set) == len(
            required_placeholders_set & exist_placeholders_1), "all required placeholders should exist"
        # losses in training steps should exist in self.placeholder_requirement_for_losses:
        losses_in_training_steps = set()
        for batch_training_steps in self.training_steps:
            for one_step_losses in batch_training_steps:
                for loss_name in one_step_losses:
                    losses_in_training_steps.add(loss_name)
        losses_in_placeholder_requirement = set()
        for loss_name in self.placeholder_requirement_for_losses.keys():
            losses_in_placeholder_requirement.add(loss_name)
        for loss_name in losses_in_training_steps:
            assert loss_name in losses_in_placeholder_requirement, \
                "losses \"%s\" in training steps should exist in self.placeholder_requirement_for_losses" % (loss_name)
        # different eval step can not have same losses, one eval step can not have same losses:
        tmp = set()
        for losses_in_one_eval_step in self.eval_steps:
            for loss_name in losses_in_one_eval_step:
                assert loss_name not in tmp, "different eval step can not have same losses, one eval step can not have same losses"
                tmp.add(loss_name)

    def check_after_build_graph(self):
        # losses and losses_to_only_watch can not have overlap:
        losses_to_optimize = self.get_items_group_by_name_from_by_id(self.losses_groups)
        losses_to_watch = self.get_items_group_by_name_from_by_id(self.losses_only_watch_groups)
        LOS = set([name for name in losses_to_optimize.keys()])
        LWS = set([name for name in losses_to_watch.keys()])
        LAS = LOS | LWS  # Union
        assert len(LAS) == len(LOS) + len(LWS)

    @abstractmethod
    def build_inferring_graph(self, group_id: int) -> NoReturn:
        pass

    @abstractmethod
    def build_training_graph(self, group_id: int) -> NoReturn:
        pass

    @abstractmethod
    def process_origin_data_for_placeholders(self, data: Dict[str, List[Any]], for_loss_n: str = None) -> Dict[
        str, List[Any]]:
        """
        :param data:
        :param for_loss_n:
        :return:
        """
        pass

    @abstractmethod
    def vars_mapping_for_loading_transfer_param(self, vars_to_store: List[tf.Variable]) -> Dict[str, str]:
        pass

    @abstractmethod
    def merge_batch_prediction_result(self, new_batch_result: Dict[str, Any],
                                      previous_result: Union[Dict[str, Any], None]):
        pass

    @abstractmethod
    def set_vocab_size(self, vocab_size: int) -> NoReturn:
        pass

    @abstractmethod
    def new_losses_are_better(self, new_losses: Dict[str, float], old_losses: Dict[str, float]) -> bool:
        pass

    def feed_dict_post_process(self, feed_dict: Dict, data_name_not_exist: Set[str]) -> Dict:
        return feed_dict

    def create_placeholders(self, group_id: int) -> NoReturn:
        """an example of meta_dict: {'input':[dtype, shape, name]}"""
        one_group_placeholders = {}
        for key in self.placeholders_meta_data.keys():
            x = self.placeholders_meta_data[key]
            one_group_placeholders[key] = tf.placeholder(dtype=x.dtype, shape=x.shape, name=key + "_%d" % group_id)
        self.placeholder_groups[group_id] = one_group_placeholders

    def get_items_group_by_name_from_by_id(self, items_group_by_parallel_id: Dict[int, Dict[str, Any]]) -> Dict[
        str, List]:
        """
        Only for transfer Dict[int, Dict[str, Any]] into Dict[str, List[Any]]
        :param items_group_by_parallel_id:
        :return:
        """
        losses_dict = {}
        for id in items_group_by_parallel_id.keys():
            one_group = items_group_by_parallel_id[id]
            for name in one_group.keys():
                if name in losses_dict:
                    losses_dict[name].append(one_group[name])
                else:
                    losses_dict[name] = one_group[name]
        return losses_dict

    def get_all_configs_in_json(self):
        # todo:how to convert a complex object to json?
        config_json = {}
        for key in self.config.keys():
            if type(self.config[key]) == PlaceholderMetaData:
                config_json[key] = self.config[key].__dict__
            else:
                config_json[key] = self.config[key]
        print(config_json)
        return json.dumps(config_json, sort_keys=True, indent=4)
