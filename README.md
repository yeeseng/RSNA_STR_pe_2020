# RSNA_STR_pe_2020

This repository contains our competition source code that won us a 14th place out of 784 teams in the RSNA STR Pulmonary Embolism Detection Challenge (https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection) 

## Solution overview
We used the "CNN+RNN" schema inspired by last year's RSNA winning solution by SeuTau (https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection.git).

![schematic](/figures/RSNA_PE_diagram.jpeg)

### Data and preprocessing
The raw data is divided into five cross-validation sets stratified by the labels provided. The raw CT data is windowed using "lung", "PE", and "mediastinal" windows as suggested by Ian Pan on this discussion post: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930

The windowing preprocessing step generates a 3 channel dataset that can be conveniently used to train models pretrained on RBG non-radiology images.

Please refer to Kaggle competition website (https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data) for instructions to download the dataset. It is too big to be included in this repository.

### Stage 1: 2D CNN classifier

For the stage 1 2D CNN model, we used resnet50 pretrained on imageNet and retrain the model on the competition data preserving the native 512x512 resolution of the CT images. The decision was driven by the observation that pulmonary embolisms can be very small and may be only a few pixels wide, thus neccesitating the high resolution. Our early experiments confirm our intuition. We found that it is better to use a smaller model with native resolutions than to use a larger model but having to downscale the images to meet the memory constraints of the GPU.

During training, we used conventional data augmentation techniques including random scale, shift, rotate, blur, add noise. One of the challenges of this competition is class imbalance as at the image level, most images do not contain PE. To combat this, we added larger class weight positive images. A trick during training that we found worked really well is to train the stage 1 CNN model on both image level labels as well as study level labels. To do this, we adjust the study labels to make it more usable on the image level. For example, a positive study may have images that are positive for PEs and images that are negative. The study labels "Left PE", "Right PE", "Central PE" can only make sense if "PE present on image" and carried over to be used as image level labels. For images where PE is not present on image, these are defaulted to False. We found that applying study level labels in this manner helped to increase the accuracy of all the labels, including the image level label "PE present on image".  

Once the stage 1 CNN model is fully trained, the last layer is peeled off so that the model outputs a vector of size 64. The feature vectors of all images in a study are what is being fed en-bloc into the stage 2 model.

Codes:
Run "step01_CNNmodel_02-CV0.ipynb" notebook for a template of our training codes.
Run "step02_extractFeatures_02-CV0.ipynb" to convert the trained model to a feature extractor for stage 2.

### Stage 2: RNN model

For stage 2 model, we used bidirectional GRU architecture. We actively tried to prevent overfitting because there are only 7000+ studies. Hence, our input size and hidden states were intentionally small. We chose GRU over other RNN cells like LSTM for the same reason. It occured to us that we can use larger architectures if we use other techniques to combat overfitting such as dropout layers. However, at the end this is a code competition with compute constraints, so we opt to keep things simple and stick with our smallish RNN models. It turned out to be a good decision when we found that we could add more RNN heads to the same CNN base model and ensemble the results to improve our accuracies.  

Codes:
Run "step03_EmbeddingModel-02_CV0_3.ipynb" notebook for a template of our training codes. 

### Submission

Despite spliting the data to five cross-validation sets, we were only able to incorporate our three best performing cross-validation models due to the compute constraints imposed by this code competition. The final public Kaggle notebook for submission can be found here https://www.kaggle.com/yeeseng/rsna-pe-submission-final-pydicom

## Installation

We use anaconda to manage our environments. Please refer to anaconda documentation available here at https://docs.anaconda.com/anaconda/install/ for installation instructions. We have attached our environment yaml file here to facilitate recreation of the environment.

```bash
conda env create --file=environment.yaml
```
