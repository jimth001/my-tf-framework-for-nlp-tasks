from TextPreprocessing.gpt_bpe_tool import get_encoder
from TextPreprocessing.ekphrasis_for_preprocess import get_text_processor
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from MyEstimator.ModelFn import PlaceholderType, PlaceholderMetaData, ModelFn
from collections import OrderedDict
from typing import List, Dict
import random
import os
import tensorflow as tf


def load_txt_corpus(path, data_type: PlaceholderType):
    """
    load data for one placeholder
    :param path:
    :param data_type:
    :return:
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if data_type == PlaceholderType.Text:
                lines.append(line.strip())
            elif data_type == PlaceholderType.TargetInt:
                lines.append(int(line.strip()))
            elif data_type == PlaceholderType.TargetFloat:
                lines.append(float(line.strip()))
            else:
                raise ValueError("Unexpected data type: %s", str(data_type))
    return lines


def load_tsv_corpus(path, placeholders: Dict[str, PlaceholderMetaData]):
    """
    load tsv data for all placeholders
    In this way, we can first check if the data and the placeholders are matched.
    :param path: the path of the tsv file
    :return: a dict whose keys are placeholders' name, and the values are the data.
    """

    def convert_type(text_data, type):
        if type == PlaceholderType.Text:
            return text_data
        elif type == PlaceholderType.TargetInt:
            return int(text_data)
        elif type == PlaceholderType.TargetFloat:
            return float(text_data)
        else:
            raise ValueError("Unexpected data type: %s", str(type))

    data = OrderedDict()
    first = True
    pls_name = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if first:
                first = False
                pls_name = line.strip().split('\t')
                #assert len(pls_name) == len(placeholders)
                #assert len(pls_name) == len(set(pls_name))
                for n in pls_name:
                    assert n in placeholders
                    data[n] = []
            else:
                ld = line.strip().split('\t')
                for i, n in enumerate(pls_name):
                    data[n].append(convert_type(ld[i], type=placeholders[n].type))
    return data


class DataStream:
    """
    To avoid confused configurations, DataStream will append EOS to each text.
    If some task do not need eos, drop it in func_for_task_specific_processing.
    """
    def __init__(self, files_base_path, placeholder_meta_data: Dict[str, PlaceholderMetaData],
                 *, func_for_task_specific_processing: ModelFn.process_origin_data_for_placeholders,
                 text2index_dictionary_path='./data/bpe_codes/', in_tsv_mode=True,
                 text_preprocessor=get_text_processor(), text2index_tool=get_encoder,
                 shuffle_each_epoch=False, round_feeding=True):
        """
        :param files_base_path: For tsv mode. it is the path of tsv file; for txt mode, it is the parent directory of txt files.
                The name of txt file is in placeholder_meta_data
        :param placeholder_meta_data:
        :param func_for_task_specific_processing: to create data for placeholders which has "Ref"
        :param text2index_dictionary_path: bpe codes path
        :param text_preprocessor: an object for preprocess, such as normalization, correcting spelling errors and so on.
                We apply a default text_preprocessor--ekphrasis.
        :param text2index_tool: an object for bpe encoding and decoding. We use gpt's bpe tool as default.
        :param shuffle_each_epoch: if true, the dataset will be shuffled at the start of each epoch.
        :param round_feeding: if true, this stream will provide data in cycling way, it means that stream will turn around
                when pointer get the end of data, and it will alaways provide enough "size" data. if false, when pointer get
                to the end, fetch data process will be ended, then the iter will be re-initilized.
        """
        assert os.path.exists(files_base_path), "path not exsit: %s" % files_base_path
        # load origin data:
        if in_tsv_mode:
            data = load_tsv_corpus(files_base_path, placeholders=placeholder_meta_data)
        else:
            data = OrderedDict()
            for n in placeholder_meta_data.keys():
                data[n] = load_txt_corpus(os.path.join(files_base_path, placeholder_meta_data[n].from_file_name),
                                          placeholder_meta_data[n].type)
        # confirm data is aligned and data size:
        dlen = [len(data[n]) for n in data.keys()]
        assert max(dlen) == min(dlen), "dataset is not aligned: %s" % str(dlen)
        self.dataset_size = max(dlen)
        # text preprocess and 2index:
        self.text_preprocessor = text_preprocessor
        self.text_index_encoder = text2index_tool(text2index_dictionary_path)
        self.append_eos = True#do not change this.
        for n in data.keys():
            if placeholder_meta_data[n].type == PlaceholderType.Text:
                text = self.text_preprocessing(data[n])
                data[n] = self.text2index(text)
        self.ori_data = data
        self.placeholder_meta_data = placeholder_meta_data
        # task specific processing:
        self.processed_data = func_for_task_specific_processing(data)
        # vars for iter:
        self.low = 0
        self.epoch = 1
        self.shuffle_each_epoch = shuffle_each_epoch
        if self.shuffle_each_epoch:
            self.shuffle_data(self.processed_data)
        # whether to turn around
        self.round_feeding=round_feeding
        # add config information for recording: todo make config inf more useful
        self.config = {}
        self.config['in_tsv_mode'] = in_tsv_mode
        self.config['text_preprocessor'] = 'ekphrasis'
        self.config['bpe_tool'] = 'gpt'

    def reset_feeding_status(self):
        self.epoch=1
        self.low=0

    def shuffle_data(self, data: Dict[str, List]):
        aligned_data = []
        for i in range(0, self.dataset_size):
            t = []
            for key in data.keys():
                t.append(data[key][i])
            aligned_data.append(t)
        random.shuffle(aligned_data)
        new_data_dict = OrderedDict()
        for i, key in enumerate(data.keys()):
            t = []
            for j in range(0, self.dataset_size):
                t.append(aligned_data[j][i])
            new_data_dict[key] = t
        return new_data_dict

    def text_preprocessing(self, data):
        if self.text_preprocessor is not None:
            return [" ".join(self.text_preprocessor.pre_process_doc(l)) for l in data]
        else:
            return data

    def text2index(self, data: List[str]):
        indexes = []
        for l in data:
            if self.append_eos:
                indexes.append(self.text_index_encoder.encode(l) + [self.text_index_encoder.eos_id])
            else:
                indexes.append(self.text_index_encoder.encode(l))
        return indexes

    def index2text(self, data: List[str]):
        text = []
        for l in data:
            if self.append_eos:
                if l[-1] == self.text_index_encoder.eos_id:
                    l = l[:-1]
            text.append(self.text_index_encoder.decode(l))
        return text

    def padding_batch(self, input_list):
        in_len = [len(i) for i in input_list]
        new_in = pad_sequences(input_list, padding='post', value=0)
        return new_in, in_len

    """def padding_for_target_mask(self, mask_list, input_len):
        batch_size = len(mask_list)
        assert batch_size == len(input_len)
        max_len = max(input_len)
        for i in range(0, batch_size):
            l = input_len[i]
            mask_list[i] = mask_list[i] + [0.0] * (max_len - l)"""

    def get_feed_dict(self, one_group_placeholders: Dict[str, tf.placeholder], size):
        feed_dict = {}
        for n in one_group_placeholders.keys():
            feed_dict[one_group_placeholders[n]] = []
        if self.round_feeding:
            while size > 0:
                if self.low + size <= self.dataset_size:
                    for n in one_group_placeholders.keys():
                        feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:self.low + size]
                    self.low += size
                    size = 0
                else:
                    for n in one_group_placeholders.keys():
                        feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:]
                    size = size - (self.dataset_size - self.low)
                    self.low = 0
                    self.epoch += 1
                    if self.shuffle_each_epoch:
                        self.shuffle_data(self.processed_data)
        else:
            for n in one_group_placeholders.keys():
                feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:self.low + size]
            self.low+=size
            if self.low>=self.dataset_size:
                self.epoch+=1
        # padding sequence:(len has been already calculated in func_for_task_specific_processing)
        for key in one_group_placeholders.keys():
            if self.placeholder_meta_data[key].type == PlaceholderType.Text or \
                    self.placeholder_meta_data[key].type == PlaceholderType.TextTargMask or \
                    self.placeholder_meta_data[key].type == PlaceholderType.TextAttnMask:
                feed_dict[one_group_placeholders[key]], _len = self.padding_batch(feed_dict[one_group_placeholders[key]])
        return feed_dict

    def get_all_configs(self):
        # todo
        pass
