# HuBMAP-pipeline
Pytorch Lightning training pipeline of semantic segmentation model for the identification of glomeruli in human kidney tissue slides.

## Training
Training is executed by running the ```run_experiments.sh``` script which calls ```swm.py``` with a default set of hyperparameters.


## Results
**DICE score**: ***0.96090***

Tissue slides:

![Tissue slides](https://i.imgur.com/RRJfIdk.png)


ground truth (top) vs. predicted binary segmentation mask (bottom)
![ground truth/predicted mask](https://i.imgur.com/RbnhUV4.png)

