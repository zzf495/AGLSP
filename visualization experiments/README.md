# Code for visualization experiments

This repository contains the codes for visualization experiments:

---

raw: The folder storing the raw images;

params: The folder storing the weight extracted by `Grad-CAM`. To facilitate the extraction of other features, we provide two codes in `/params/`  (`resnet_feature_extraction.py` and `myResnet.py`), which transforms the input raw data into the desired weight matrix. The codes of `Grad-CAM` can be downloaded [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations).

data: The folder storing the data extracted by ResNet50 (note: the features should be consistent with that in `Grad-CAM`.

algorithm_on_Office31: The learned projection matrix of compared algorithms. The projected features (subspaces) are preserved manually by running the codes on the task.

## Code files (matlab implementation)

runVisualization.m: The demo that visualization the results of AGE-CS and AGLSP on task transferring from "amazon" to "dslr" on Office31 (The data for running this code can be downloaded from [here](), and it needs to be extracted to the root directory).

