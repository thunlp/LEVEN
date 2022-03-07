# LEVEN
Dataset and source code for ACL 2022 Findings paper "LEVEN: A Large-Scale Chinese Legal Event Detection Dataset" .

## Overview

The dataset can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6e911ff1286d47db8016/) or [Google Drive](https://drive.google.com/drive/folders/1VGD0h365kegTqGEyLr24SJtJUUoZIt20?usp=sharing). 

### Large Scale

LEVEN is the largest Legal Event Detection dataset and the largest Chinese Event Detection dataset. Here is a comparison between the scale of LEVEN and other datasets. 

![tab1](./pic/tab1.jpg)

Datasets denoted with * are not publicly available, and – means the value is not accessible

### High Coverage

LEVEN contains 108 event types in total, including 64 charge-oriented events and 44 general events. Their distribution is shown below.

![tab2](./pic/tab2.jpg)

The LEVEN event schema has a sophisticated hierarchical structure, which is shown [here](#Event Schema).  

## Leader Board

To get the test results, you can submit your predictions to our CodaLab competition (link is coming soon). 

## Experiments

The source codes for the experiments are included in the [Baselines](./Baselines) and [Downstreams](./Downstreams) folder.

​	The Baselines folder includes [DMCNN](./Baselines/DMCNN), [BiLSTM](./Baselines/BiLSTM), [BiLSTM+CRF](./Baselines/BiLSTM+CRF), [BERT](./Baselines/BERT), [BERT+CRF](./Baselines/BERT+CRF) and [DMBERT](./Baselines/DMBERT).

​	The Downstreams folder includes [Legal Judgment Prediction](./Downstreams/LJP) and [Similar Case Retrieval](./Downstreams/SCR).

### Baselines

We implement six competitive [Baselines](./Baselines) and their performances are as follows.

![tab3](./pic/tab3.jpg)

### Downstream Tasks

We also explore the use of LEVEN on two [Downstreams](./Downstreams). We simply use event as side information to promote the performance of [Legal Judgment Prediction](./Downstreams/LJP) and [Similar Case Retrieval](./Downstreams/SCR). 

The experiment results for Legal Judgment Prediction are shown below.

![tab4](./pic/tab4.jpg)

The experiment results for Similar Case Retrieval are shown below.

![tab5](./pic/tab5.jpg)

## Event Schema

The Chinese event schema is shown below. Please check our paper for the English version.

![schema](./pic/schema-zh.png)

## Citation

If these data and codes help you, please cite this paper.