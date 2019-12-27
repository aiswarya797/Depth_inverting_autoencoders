# Depth_inverting_autoencoders
## Motivation & Aim

This repository contains the work done as part of my undergraduate thesis at The Mortimer B. Zuckerman Mind Brain Behavior Institute, Columbia University, New York, from August 2019 - December 2019.

The aim of the thesis is to investigate the following research question : **How does recurrent processing help an autoencoder in image reconstruction, when the objects are under occlusion in the image?**  Our hypothesis is that recurrent processing in networks help in image recognition under partial occlusion. In order to do this, the task of the autoencoder is to understand the identity of the digits, understand the relative depth (or the order in which the digits are presented), invert the order and reconstruct the digits as output. 

## Code
There are two folders namely 'Dataset' and 'Architectures'. **Datasets** is used to create the images for the training and test set; and was based on and adapted from Spoerer et al. 2017. **Architectures** has the code for the autoencoder model.

## Results
All the experiments and results can be found in the thesis report. 

## Poster Presentation
The results of this work were showcased at the **17th NVP Winter Conference on Brain and Cognition that took place in Hotel Zuiderduin, Egmond aan Zee, the Netherlands, from December 19-21, 2019.**

## Ongoing work
It is clearly seen that the recurrent connections give really good results when compared to model without recurrent connections. But it was noticed that even a linear shallow autoencoder (only one linear layer in the encoder and the decoder respectively) could reconstruct the input image properly (as mentioned in the thesis report). So we conducted PCA on the data used and we observed that a small number of principal components could explain almost all of the variance in the data. Hence, our next direction is to change the dataset, where in the data is more difficult and then see how the shallow model performs against the deep model.

