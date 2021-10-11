# The Rich Get Richer:<br> Disparate Impact of Semi-Supervised Learning


## Preprocess file of the dataset used in implicit sub-populations:  <br>(Demographic groups: race and gender)
The following code will pre-process the jigsaw dataset and return xxx.

#### Step-1:
Download the jigsaw dataset: *identity_individual_annotations.csv* from

https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.

#### Step-2:
```
python preprocecss_jiasaw_toxicity_gender_and_race_balanced.py
```

## Implementation of SSL methods
Please follow the official implementations of MixMatch, MixText, and UDA.


[1] https://github.com/google-research/mixmatch

[2] https://github.com/GT-SALT/MixText

[3] https://github.com/google-research/uda