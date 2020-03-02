from DeepComponents.GPTEncoder import Encoder
from DeepComponents.TransformerBlock import default_hparams
from MyEstimator.DataStream import DataStream
from MyEstimator.ModelWrapper import multi_gpu_trainer
from Models.GPTModel import GPTModel
from TextPreprocessing.gpt_bpe_tool import get_encoder, Encoder


def train():
    model = GPTModel()
    trainer = multi_gpu_trainer(device_id=[0], model_fn=model)
    trainer.training(train_data_path='./data/my_s2s_test_data/train.tsv',
                     dev_data_path='./data/my_s2s_test_data/dev.tsv',
                     ckpt_dir='./data/my_s2s_models_adam/',
                     learning_rate=1e-4, batch_size=64, mini_batch=8,
                     total_steps=5000, eval_per_n_steps=100, max_to_save=1,
                     early_stop_steps=500,
                     pretrained_ckpt='./data/gpt2_pre_trained_model')


def test():
    model = GPTModel()
    model.config['decode_length'] = 60
    model.config['beam_size'] = 8
    model.config['decode_alpha'] = 0.6
    data_stream = DataStream('./data/my_s2s_test_data/test2.tsv', placeholder_meta_data=model.config['placeholders'],
                             func_for_task_specific_preprocessing=model.process_origin_data_for_placeholders,
                             shuffle_each_epoch=False, round_feeding=False, in_tsv_mode=True)
    model.config['eos_id'] = data_stream.text_index_encoder.eos_id
    my_estimator = multi_gpu_trainer(device_id=[0], model_fn=model)
    result = my_estimator.inferring(data_stream=data_stream,
                                    ckpt_dir='./data/my_s2s_models_adam/',
                                    mini_batch=8, logging=True)
    result = result['beam_search_seqs']
    for one in result:
        print(data_stream.text_index_encoder.decode(
            model.parse_out_idx(one[0], eos_id=data_stream.text_index_encoder.eos_id)))


test()
print('test finished')
