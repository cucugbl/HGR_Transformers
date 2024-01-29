# 2023-HGR5-CNN_transformer
Code of the paper
***HGREncoder: Enhancing Real-Time Hand Gesture Recognition with
Transformer Encoder - A Comparative Study**** 


# Instructions
All these instructions are identical for the CNN or the CNN-transformer model. 
To choose the desired model run the corresponding script from either the folder CNN/ or CNN-transformer/. 
To change between doing or not doing post-processing, change the corresponding flag in the **Shared.m** script.

## Configurations
1. Most of the models hyper-parameter can be configured in the **Shared.m** script.
1. Neural network architecture and trianing parameters are defined in the **modelCreation.m** script.

## Requirements
1. Download the [EMG-EPN-612 dataset](https://laboratorio-ia.epn.edu.ec/es/recursos/dataset/2020_emg_dataset_612) and paste it in **EMG-EPN-612 dataset/**. 
2.Download the eval_HGRresponses-master (https://github.com/laboratorioAI/eval_HGRresponses)

## Training
1. Create the datastores running the script: **spectrogramDatasetGeneration.m**. 
1. Train the model running the script: **modelCreation.m**.
* Trained models are saved by date in the folder **Models/** or **Modelstransformer/**.
1. Evaluate training and validation recognition accuracy running the script: **modelEvaluation.m**.

## Testing
1. Evaluate on the testing subset of the EMG-EPN-612 dataset by running the script: **testDataEvaluation.m**.
* To change the model to be evaluated, change the variable *modelFileName* in the corresponding script.
* The script will generate a **responses.json** file with the predictions in the folder "*model*/Test-Data/".
1. If desired, submit the **responses.json** file to the public online evaluator. 



<!-- Execution -->

# Abstract

Hand gestures represent a natural form of communication and device control. In the field of
Hand Gesture Recognition (HGR), Electromyography (EMG) is used to detect the electrical
impulses that muscles emit when a movement is generated. Currently, there are several
HGR models that use EMG to predict hand gestures. However, most of these models have
limited performance in practical applications. This study addresses this issue by using transformers to improve performance and mitigate ambiguity in HGR results. The architecture of
our model is composed of a Convolutional Neural Network (CNN), a positional encoding
layer and the transformer encoder. To obtain a generalizable model, the EMG-EPN-612
dataset was used. This dataset contains records of 612 individuals. The results were compared with previous research that used CNN, transformer and transformers. The findings of this
research reached a classification accuracy of 95.25 ±4.9% and a recognition accuracy of
91.73 ±8.19% 




# Authors
LUIS GABRIEL MACÍAS SANTILLÁN
luis.macias@epn.edu.ec
DIRECTOR: PhD. MARCO ENRIQUE BENALCÁZAR PALACIOS
marco.benalcazar@epn.edu.ec
CO-DIRECTOR: PhD. LORENA ISABEL BARONA LÓPEZ
lorena.barona@epn.edu.ec


# Reference
Code of the paper: 

HGREncoder: Enhancing Real-Time Hand Gesture Recognition with
Transformer Encoder - A Comparative Study

DESARROLLO DE UN MODELO DE RECONOCIMIENTO DE
CINCO GESTOS DE LA MANO DERECHA USANDO REDES
NEURONALES TRANSFORMERS


