from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from MyEstimator.ModelFn import PlaceholderType, PlaceholderMetaData, ModelFn
from collections import OrderedDict
from typing import List, Dict, Callable, Set, Union
import random
import os
import tensorflow as tf
from TextPreprocessing.TextIndexTranslator import TextIndexTranslator
from TextPreprocessing.TextPreprocessor import TextPreprocessor
from MyEstimator.utils import flatten_nested_list


def load_txt_corpus(path, data_type: PlaceholderType):
    """
    load txt data for one placeholder
    :param path:
    :param data_type: A PlaceholderType object. Support str, int and float.
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


def load_tsv_corpus(path, placeholders: Dict[str, PlaceholderMetaData]) -> Dict[str, List[str]]:
    """
    load tsv data for all placeholders (RECOMMENDED).
    tsv format:(\t split)
    placeholder_name1   placeholder_name2   ...
    sample_1_data_1  sample_1_data_2    ...
    sample_2_data_1  sample_2_data_2    ...
    ......
    format ended
    :param path: the path of the tsv file
    :return: {placeholder_name1:List_1, ...}: A dict whose keys are placeholders' name, and the values are the data.
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
                for n in pls_name:
                    assert n in placeholders, '%s is not a placeholder\'s name' % n
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
                 *, func_for_task_specific_preprocessing: ModelFn.process_origin_data_for_placeholders,
                 text_preprocessor: Union[TextPreprocessor, None], text2index_tool: TextIndexTranslator,
                 in_tsv_mode=True,
                 shuffle_each_epoch=False, round_feeding=True):
        """
        :param files_base_path: For tsv mode. it is the path of tsv file; for txt mode, it is the parent directory of txt files.
                The name of txt file is in placeholder_meta_data
        :param placeholder_meta_data:
        :param func_for_task_specific_preprocessing: to create data for placeholders which has "Ref"
        :param text2index_dictionary_path: bpe codes path
        :param text_preprocessor: an object for preprocess, such as normalization, correcting spelling errors and so on.
                Should implement the interface: pre_process_doc(doc:str)->str, otherwise you should modify self.text_preprocessing
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
        self.text_index_encoder = text2index_tool
        self.append_eos = True  # do not change this. If you do not need eos, drop it in your modelfn.func_for_task_specific_processing
        for n in data.keys():
            if placeholder_meta_data[n].type == PlaceholderType.Text:
                text = self.text_preprocessing(data[n])
                data[n] = self.text2index(text)
        self.ori_data = data
        self.placeholder_meta_data = placeholder_meta_data
        # task specific preprocessing:
        self.processed_data = func_for_task_specific_preprocessing(data)
        # vars for iter:
        self.low = 0
        self.epoch = 1
        self.shuffle_each_epoch = shuffle_each_epoch
        if self.shuffle_each_epoch:
            self.shuffle_data(self.processed_data)
        # whether to turn around
        self.round_feeding = round_feeding
        # add config information for recording: todo make config inf more useful
        self.config = {}
        self.config['in_tsv_mode'] = in_tsv_mode
        self.config['text_preprocessor'] = self.text_preprocessor.name if self.text_preprocessor is not None else 'None'
        self.config['bpe_tool'] = self.text_index_encoder.name

    def reset_feeding_status(self):
        self.epoch = 1
        self.low = 0

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

    def text2index(self, data: List[str]) -> List[List[int]]:
        indexes = []
        for l in data:
            if self.append_eos:
                indexes.append(self.text_index_encoder.encode(l) + [self.text_index_encoder.eos_id])
            else:
                indexes.append(self.text_index_encoder.encode(l))
        return indexes

    def index2text(self, data: List[List[int]]) -> List[str]:
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

    def get_feed_dict(self, one_group_placeholders: Dict[str, tf.placeholder], size,
                      op_name_to_run_and_target_data_name: Dict[str, List[str]],
                      modelfn_post_process: Callable[[Dict, Set[str]], Dict]) -> Dict:
        """
        An example for helping understand what self.get_feed_dict do:
        :param one_group_placeholders: {data1:tf.placeholder,data2:tf.placeholder,data3:tf.placeholder,
                                         batch_size:tf.placeholder}
        :param size: 10
        :param op_name_to_run_and_target_data_name: {loss1:[data1,data2,batch_size]}
        self.processed_data={data1:[xx,...],data2:[xx,...],data3:[xx,...]}
        self.placeholder_meta_data={data1:PlaceholderMetaData,data2:PlaceholderMetaData,data3:PlaceholderMetaData,
                                    batch_size:PlaceholderMetaData}
        Process steps:
        1. squeeze needed data names to a set: data_names_to_fetch={data1,data2,batch_size}
        2. split data_names_to_fetch into two sets: data_included={data1,data2}, data_excluded={batch_size}
        3. check if elements in data_excluded are all batch information
        4. fetch_data for data_included
        :return: {data1:[x1,...,x10],data2:[x1,...,x10]}
        """
        feed_dict = {}
        size_to_fetch = size
        data_names_to_fetch = [op_name_to_run_and_target_data_name[key] for key in
                               op_name_to_run_and_target_data_name.keys()]
        data_names_to_fetch = set([x for x in flatten_nested_list(data_names_to_fetch)])
        data_included = set()
        data_excluded = set()
        for name in data_names_to_fetch:
            if name in self.processed_data:
                data_included.add(name)
            else:
                data_excluded.add(name)
                assert self.placeholder_meta_data[name].type == PlaceholderType.SpecialBatchInformation or \
                       self.placeholder_meta_data[name].type == PlaceholderType.BatchSize, \
                    'type of %s is %s, should be %s or %s, or should be in self.processed_data' \
                    % (name, self.placeholder_meta_data[name].type,
                       PlaceholderType.BatchSize, PlaceholderType.SpecialBatchInformation)
        for n in data_included:
            feed_dict[one_group_placeholders[n]] = []
        if self.round_feeding:
            while size > 0:
                if self.low + size <= self.dataset_size:
                    for n in data_included:
                        feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:self.low + size]
                    self.low += size
                    size = 0
                else:
                    for n in data_included:
                        feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:]
                    size = size - (self.dataset_size - self.low)
                    self.low = 0
                    self.epoch += 1
                    if self.shuffle_each_epoch:
                        self.shuffle_data(self.processed_data)
        else:
            for n in data_included:
                feed_dict[one_group_placeholders[n]] += self.processed_data[n][self.low:self.low + size]
            self.low += size
            if self.low >= self.dataset_size:
                self.epoch += 1
        # padding sequence:(len has been already calculated in func_for_task_specific_processing)
        for key in data_included:
            if self.placeholder_meta_data[key].type == PlaceholderType.Text or \
                    self.placeholder_meta_data[key].type == PlaceholderType.TextTargMask or \
                    self.placeholder_meta_data[key].type == PlaceholderType.TextAttnMask:
                feed_dict[one_group_placeholders[key]], _len = self.padding_batch(
                    feed_dict[one_group_placeholders[key]])
        # process for batch information(Only batch size now, more task specific process should be done in modelfn.feed_dict_post_process):
        for key in data_excluded:
            if self.placeholder_meta_data[key].type == PlaceholderType.BatchSize:
                if self.low >= self.dataset_size:
                    batch_size = size_to_fetch - (self.dataset_size - self.low)
                else:
                    batch_size = size_to_fetch
                feed_dict[one_group_placeholders[key]] = batch_size
        return modelfn_post_process(feed_dict, data_excluded)

    def get_all_configs(self):
        # todo
        pass
