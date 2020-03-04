# My-TF-Framework-for-NLP-Tasks (for Tensorflow 1.X)
This project aims to help people implement Tensorflow model 
pipelines quickly for different nlp tasks.

## Features
1. Multi-GPU Training (Data Parallel)
2. Gradient Accumlation  
3. Flexible ModelFn which easy to define. Most time, except to define the Graph, you should almost only need to complete within 100 line codes to finish your model.  
3.1. Flexible Losses, Preditions. You can define different losses, predictions for multi-task learning.  
3.2. Flexible Running Control. You can fully and easily to control the training, inferring steps, such as to define a training cycle which has different losses to optimize for each step.
With this feature, you can finish some "training-complex-model" (such as GAN, compared to a Classifier) easily and you do not need to implement a big function to control your training process.

##Kernel Classes Introduction
###1. ModelFn  
The Base Class for your own models.  
Define **Inputs**, **Losses to optimize**, **Losses to only watch**, **Predictions**, **Running Control** and the **TF Graph**.  
A model could have n Inputs, m Losses to optimize, k Losses to only watch, t Predictions to output.  
Every input, loss, prediction should have a name.  
Each **Loss** requires x **Inputs**, which can be represented as {loss_name:[input_name,...]}.
Each **Prediction** requires x **Inputs**, which can be represented as {prediction_name:[input_name,...]}.  

Different losses, predictions, optimize_ops may have conflicts so sometimes you can not running once to fetch all of them.
Although this is almost not to happen, we also provide a strategy to deal with it.
In eval stage, we apply an list named eval_steps to define how to eval all losses in n steps.
    
    n=len(eval_steps)
    eval_steps=[step1,step2,...stepn]
    step_i is a list of losses
    You can put conflict losses into different steps.
    Notice that use more than 1 eval steps will cause additional computational overhead,
    so you should use it only when necessary.
In training stage, similarly but differently, we also provide a "training_steps" list, where:  
                
    training_steps=[batch_1_step,batch_2_step,...]
    batch_1_step=[step1,step2,...]
    step1=[loss1,...] ...
    In training stage:
    for one_batch_steps in training_steps:
        produce one batch data
            for losses in one_batch_steps:
                split losses into can_be_optimized and only_watch
                then train(can_be_optimized,data) and fetch(only_watch,data)
In training stage, ModelWrapper create a train_op for each optimized loss.  
If a loss is in self.losses_groups, ModelWrapper will run the corresponding train_op 
and fetch the loss value, otherwise ModelWrapper will only fetch the loss value to display.  

In prediction stage, there is a list named "predicting_steps", similar to "eval_steps".

###2. ModelWrapper
Implement multi-gpu running and gradient accumlation. 

###3. DataStream
Load data, preprocess and generate batch data.

##Get Started
We apply a simple example in Models.GPTModel, and test functions are in main.py.  

