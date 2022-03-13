# The Rich Get Richer:<br> Disparate Impact of Semi-Supervised Learning

## Implementation of SSL methods
Please follow the official implementations of MixMatch, MixText, and UDA.


[1] https://github.com/google-research/mixmatch

[2] https://github.com/GT-SALT/MixText

[3] https://github.com/google-research/uda





## Experiments on CIFAR
Our experiments are run on NVIDIA RTX A5000. The recommended version of Pytorch is 1.10.1+cu113. You may change the torch version based on your GPUs.
### Requirements:
- torch==1.10.1+cu113 
- torchvision==0.11.2+cu113 
- tensorboardx
- matplotlib
- psutil
- requests
- progress
- numpy


### Quick Start
We provide an example of our implementation of MixMatch based on the open-sourced code (pytorch version, unofficial): https://github.com/YU1ut/MixMatch-pytorch.

Run the following code:
```shell
python3 train.py --gpu 0 --n-labeled <labeled_size>  --dataset <cifar10 or cifar100> --sample Random --train_mode <small or ssl> --out ./path/to/results/
# small: The baseline method. Train with a small labeled dataset.
# ssl: Train with MixMatch
```

Alternatively, you can simply run:
```shell
bash ./run_c10_mixmatch.sh # cifar-10 experiments
bash ./run_c100_mixmatch.sh # cifar-100 experiments
```

### Plot the results:
We provide the following example to reproduce Figure 3 (CIFAR-100 part):
```shell
python3 plot_c100.py
```





## Experiments on NLP datasets
For SSL parts, you can follow the official implementations of [MixText](https://github.com/GT-SALT/MixText) and [UDA](https://github.com/google-research/uda).
### Preprocess file of the dataset used in implicit sub-populations:  <br>(Demographic groups: race and gender)
The following code will pre-process the jigsaw dataset and return train/test dataset files including demographic groups information.

#### Step-1:
Download the jigsaw dataset: *identity_individual_annotations.csv* from

https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.

#### Step-2:
```
python preprocecss_jiasaw_toxicity_gender_and_race_balanced.py
```

