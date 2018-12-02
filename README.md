# Social Relation Recognition in TensorFlow

This repo is the TensorFlow implementation of CVPR 2017 Paper [A Domain Based Approach to Social Relation Recognition](https://arxiv.org/pdf/1704.06456.pdf) by [Qianru Sun](https://www.comp.nus.edu.sg/~sunqr/), [Mario Fritz](https://scalable.mpi-inf.mpg.de/), and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/).

<p align="center">
    <img src="https://github.com/Y2L/social_relation_tensorflow/blob/master/docs/framework.png" width="600"/>
</p>

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#Dataset)
* [Examples](#examples)
* [Citation](#citation)

## Introduction

Social relations are the foundation of human daily life. Developing techniques to analyze such relations from visual data bears great potential to build machines that better understand us and are capable of interacting with us at a social level. Previous investigations have remained partial due to the overwhelming diversity and complexity of the topic and consequently have only focused on a handful of social relations. In this paper, we argue that the domain-based theory from social psychology is a great starting point to systematically approach this problem. The theory provides coverage of all aspects of social relations and equally is concrete and predictive about the visual attributes and behaviors defining the relations included in each domain. We provide the first dataset built on this holistic conceptualization of social life that is composed of a hierarchical label space of social domains and social relations. We also contribute the first models to recognize such domains and relations and find superior performance for attribute based features. Beyond the encouraging performance of the attribute based approach, we also find interpretable features that are in accordance with the predictions from social psychology literature. Beyond our findings, we believe that our contributions more tightly interleave visual recognition and social psychology theory that has the potential to complement the theoretical work in the area with empirical and data-driven models of social life.

## Installation

### Requirements

In order to run this repo, we advise you to install python 2.7 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instrucation on their offical website:
[https://www.anaconda.com/download/](https://www.anaconda.com/download/)

```
conda create --name vqa python=3
source activate vqa
conda install pytorch torchvision cuda80 -c soumith
```
