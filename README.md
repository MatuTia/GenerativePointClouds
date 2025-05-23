# Dynamic Style-TreeGAN

This is the implementation of the generative model Dynamic Style-TreeGAN, for the generation of 3D point clouds with
quasi-uniform distribution and limited computational costs.
While existing GAN models usually do not focus on the design of the discriminator, in [1] we propose a novel
discriminator based on dynamic graph convolutional networks that does not require a priori information on input data
connectivity.

[1] Carlotta Giannelli, Sofia Imperatore, Mattia Matucci, Matteo Paiano. 3D point cloud generation for surface
representation. In Scientific Machine Learning Emerging Topics 2024 special volume in the SEMA-SIMAI series.

This repository contains the code and the data set used for training and testing in that publication.

## Notes

The implementation use [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) and can thus be easily
integrated into existing codes, other requirements are specified in [GAN.yaml](GAN.yaml).

[generator.py](generator.py) contains the implementation of the generator architectures

[discriminator.py](discriminator.py) contains the novel discriminator architecture characterized by a dynamic graph
convolutional operator

[loss.py](loss.py) contains the loss used for training the WassersteinGAN model

[data.py](data.py) contains the dataset structure

[train.py](train.py) defines and performs the training of the model

[test.py](test.py) produces and plots the generated samples

[metric.py](metric.py) contains the definition of the metrics, MMD and JSD

[metric_evaluation.py](metric_evaluation.py) evaluates the metrics on sampled clouds

[boundary_detection.py](boundary_detection.py) implements the boundary detection for a given point cloud

[post_process.py](post_process.py) implements the perimeter detection, and its parametrization
