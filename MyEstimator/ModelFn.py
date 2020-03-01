from abc import ABCMeta, abstractmethod
from tensorflow.python.util import nest
import tensorflow as tf
from collections import OrderedDict
from enum import Enum
import json
from typing import List,Dict,Any

class PlaceholderType(Enum):
    Text = 1
    TargetInt = 2
    TargetFloat = 3
    TextLength = 4
    TextAttnMask = 5
    TextTargMask = 6
    BatchSize = 7

class PlaceholderMetaData():
    def __init__(self,type:PlaceholderType,dtype,shape,from_file_name=None,Ref=None):
        self.type=type
        self.Ref=Ref
        self.shape=shape
        """There are two ways to load data:
        1.Read data for each placeholder from each from_file_name.
        2.Read all data from one tsv file. The first line of the tsv must be the names of corresponding placeholders.
        """
        self.from_file_name=from_file_name
        self.dtype=dtype

class ModelFn:
    def __init__(self):
        self.placeholder_groups:Dict[int,Dict[str,tf.placeholder]]=OrderedDict()
        self.losses_groups:Dict[int,Dict[str,any]] = OrderedDict()
        self.config = {}

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def build_inferring_graph(self,group_id:int):
        pass

    @abstractmethod
    def build_training_graph(self,group_id:int):
        pass

    @abstractmethod
    def process_origin_data_for_placeholders(self, data: Dict[str, List[Any]], for_loss_n:str=None) -> Dict[str, List[Any]]:
        """

        :param data:
        :param for_loss_n:
        :return:
        """
        pass

    @abstractmethod
    def vars_mapping_for_loading_transfer_param(self,vars_to_store:List[tf.Variable])->Dict[str,str]:
        pass

    @abstractmethod
    def merge_batch_prediction_result(self, new_batch_result: Dict[str, Any], previous_result: Dict[str, Any] or None):
        pass

    @abstractmethod
    def set_vocab_size(self,vocab_size:int):
        pass

    @abstractmethod
    def new_losses_are_better(self,new_losses:List,old_losses:List, losses_name:List):
        pass

    def create_placeholders(self,group_id:int):
        """an example of meta_dict: {'input':[dtype, shape, name]}"""
        one_group_placeholders={}
        for key in self.config['placeholders'].keys():
            x=self.config['placeholders'][key]
            one_group_placeholders[key]=tf.placeholder(dtype=x.dtype, shape=x.shape, name=key+"_%d"%group_id)
        self.placeholder_groups[group_id]=one_group_placeholders

    def get_all_losses(self)->Dict[str,List]:
        losses_dict={}
        for key in self.losses_groups.keys():
            one_group=self.losses_groups[key]
            for k in one_group.keys():
                if k in losses_dict:
                    losses_dict[k].append(one_group[k])
                else:
                    losses_dict[k]=one_group[k]
        return losses_dict


    def get_all_configs_in_json(self):
        #todo:how to convert a complex object to json?
        config_json={}
        for key in self.config.keys():
            if type(self.config[key])==PlaceholderMetaData:
                config_json[key]=self.config[key].__dict__
            else:
                config_json[key]=self.config[key]
        print(config_json)
        return json.dumps(config_json, sort_keys=True, indent=4)

