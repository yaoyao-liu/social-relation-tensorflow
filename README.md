# Social Relation Recognition using TensorFlow

This repo is the TensorFlow implementation of CVPR 2017 Paper [A Domain Based Approach to Social Relation Recognition](https://arxiv.org/pdf/1704.06456.pdf) by [Qianru Sun](https://www.comp.nus.edu.sg/~sunqr/), [Mario Fritz](https://scalable.mpi-inf.mpg.de/), and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/).

<p align="center">
    <img src="https://github.com/Y2L/social_relation_tensorflow/blob/master/docs/framework.png" width="600"/>
</p>

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#Dataset)
* [Usage](#usage)
* [Citation](#citation)

## Introduction

Social relations are the foundation of human daily life. Developing techniques to analyze such relations from visual data bears great potential to build machines that better understand us and are capable of interacting with us at a social level. Previous investigations have remained partial due to the overwhelming diversity and complexity of the topic and consequently have only focused on a handful of social relations. In this paper, we argue that the domain-based theory from social psychology is a great starting point to systematically approach this problem. The theory provides coverage of all aspects of social relations and equally is concrete and predictive about the visual attributes and behaviors defining the relations included in each domain. We provide the first dataset built on this holistic conceptualization of social life that is composed of a hierarchical label space of social domains and social relations. We also contribute the first models to recognize such domains and relations and find superior performance for attribute based features. Beyond the encouraging performance of the attribute based approach, we also find interpretable features that are in accordance with the predictions from social psychology literature. Beyond our findings, we believe that our contributions more tightly interleave visual recognition and social psychology theory that has the potential to complement the theoretical work in the area with empirical and data-driven models of social life.

## Installation

### Requirements

In order to run this repo, we advise you to install python 2.7 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instrucation on their offical website:
[https://www.anaconda.com/download/](https://www.anaconda.com/download/)

```Bash
conda create --name tensorflow_1.3.0_gpu python=2.7
source activate tensorflow_1.3.0_gpu
pip install --ignore-installed --upgrade https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
```

Then clone the repo install the requirements:

```Bash
git clone https://github.com/Y2L/social_relation_tensorflow.git 
cd social_relation_tensorflow
pip install scipy
pip install tqdm
```

## Dataset

### Image resource.
The image resource [People in Photo Album (PIPA)](https://people.eecs.berkeley.edu/~nzhang/piper.html) was collected from Flickr photo albums for the task of person recognition. Photos from Flickr cover a wide range of social situations and are thus a good starting point for social relations. The same person often appears in different social scenarios and interacting with different people which make it ideal for our purpose. Identity information is used for selecting person pairs and defining train-validation-test splits. In summary, PIPA contains 37,107 photos with 63,188 instances of 2,356 identities. We extend the dataset by 26,915 person pair annotations for social relations.
