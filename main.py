from DeepComponents.GPTEncoder import Encoder
from DeepComponents.TransformerBlock import default_hparams
from MyEstimator.DataStream import DataStream
from MyEstimator.TrainingWrapper import multi_gpu_trainer
from Models.GPTModel import GPTModel
from TextPreprocessing.gpt_bpe_tool import get_encoder,Encoder

model=GPTModel()
trainer=multi_gpu_trainer(device_id=[0],model_fn=model)
trainer.training(train_data_path='./data/family_relationship/train.tsv',
                 dev_data_path='./data/family_relationship/dev.tsv',
                 ckpt_dir='./data/formality_models_adam/',
                 learning_rate=1e-4,batch_size=64,mini_batch=8,
                 total_steps=5000,eval_per_n_steps=100,max_to_save=1,
                 early_stop_steps=500,
                 pretrained_ckpt='./data/gpt2_pre_trained_model')

print('test finished')

