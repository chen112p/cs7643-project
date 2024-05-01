# cs7643-project
## Introduction
This is repo stores the final project for CS7643 Deep Learning 2024 Winter by Shan Tie, Nameer Rehman and Junting Chen. In this project, we want to try different approaches on classifying hateful tweets towards immigrant and women. We applied 4 approaches, includes a hyperparamter tuned RNN, RoBERTa based classifier, LoRA fine-tuned RoBERTa classifier, and QLoRA fine-tuned RoBERTa classifier. Trained models are saved on a separated drive. If interested, please contact jchen3191@gatech.edu.

## Model training
python main.py -config MODEL_CONFIG


## Model inferencing
First copy a trained model to saved_models/. 
python inference_llm.py -device cuda -modelname MODEL_NAME
