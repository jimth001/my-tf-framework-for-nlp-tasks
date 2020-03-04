from DeepComponents.GPTEncoder import Encoder
from DeepComponents.TransformerBlock import default_hparams
from MyEstimator.DataStream import DataStream
from MyEstimator.ModelWrapper import ModelEstimator
from Models.GPTModel import GPTModel
from TextPreprocessing.gpt_bpe_tool import get_encoder, Encoder


def train():
    data_path = './data/my_s2s_test_data/train2.tsv'
    model = GPTModel()
    model.check_after_init()
    text2index_dictionary_path = './data/bpe_codes/'
    bpe_tool = get_encoder(text2index_dictionary_path)
    train_stream = DataStream(data_path, placeholder_meta_data=model.placeholders_meta_data,
                              func_for_task_specific_preprocessing=model.process_origin_data_for_placeholders,
                              text_preprocessor=None,
                              text2index_tool=bpe_tool,
                              shuffle_each_epoch=True, round_feeding=True, in_tsv_mode=True)
    dev_stream = DataStream(data_path, placeholder_meta_data=model.placeholders_meta_data,
                            func_for_task_specific_preprocessing=model.process_origin_data_for_placeholders,
                            text_preprocessor=None,
                            text2index_tool=bpe_tool,
                            shuffle_each_epoch=False, round_feeding=False, in_tsv_mode=True)
    trainer = ModelEstimator(device_id=[0], model_fn=model)
    trainer.training(train_stream, dev_stream,
                     ckpt_dir='./data/my_s2s_models_test/',
                     learning_rate=1e-4, batch_size=16, mini_batch=8,
                     total_steps=100, eval_per_n_steps=1, max_to_save=1,
                     early_stop_steps=500,
                     pretrained_ckpt='./data/gpt2_pre_trained_model/')


def test():
    model = GPTModel()
    model.config['decode_length'] = 60
    model.config['beam_size'] = 8
    model.config['decode_alpha'] = 0.6
    text2index_dictionary_path = './data/bpe_codes/'
    bpe_tool = get_encoder(text2index_dictionary_path)
    data_stream = DataStream('./data/my_s2s_test_data/test2.tsv', placeholder_meta_data=model.placeholders_meta_data,
                             func_for_task_specific_preprocessing=model.process_origin_data_for_placeholders,
                             text_preprocessor=None,
                             text2index_tool=bpe_tool,
                             shuffle_each_epoch=False, round_feeding=False, in_tsv_mode=True)
    model.config['eos_id'] = data_stream.text_index_encoder.eos_id
    model.check_after_init()
    my_estimator = ModelEstimator(device_id=[0], model_fn=model)
    result = my_estimator.inferring(data_stream=data_stream,
                                    ckpt_dir='./data/my_s2s_models_adam/',
                                    mini_batch=8, logging=False)
    result = result['pred_seq']
    for one in result:
        print(data_stream.text_index_encoder.decode(
            model.parse_out_idx(one[0], eos_id=data_stream.text_index_encoder.eos_id)))


def test_build_training_inferring_graph_simultaneously():
    model = GPTModel()
    model.config['decode_length'] = 60
    model.config['beam_size'] = 8
    model.config['decode_alpha'] = 0.6
    model.config['eos_id'] = 50256
    my_estimator = ModelEstimator(device_id=[0], model_fn=model)
    model.set_vocab_size(50257)
    my_estimator.build_data_parallel_training_graph(allow_gradient_accumulation=True)
    my_estimator.build_data_parallel_inferring_graph()
    my_estimator.create_session_init_and_print_all_vars(max_to_save=1,
                                                        pretrained_ckpt='./data/gpt2_pre_trained_model/',
                                                        logging=False)


if __name__ == '__main__':
    train()
    print('test finished')
