# LEVEN
Dataset and source code for ACL 2022 Findings paper "LEVEN: A Large-Scale Chinese Legal Event Detection Dataset".

## Dataset Overview

The dataset (ver. 1.0) can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6e911ff1286d47db8016/) or [Google Drive](https://drive.google.com/drive/folders/1VGD0h365kegTqGEyLr24SJtJUUoZIt20?usp=sharing). 

The data format is introduced in [this document](DataFormat.md).

**Data Scale**

![image-20220306230805902](C:\Users\leoyao\AppData\Roaming\Typora\typora-user-images\image-20220306230805902.png)

## Experiment

The performances of various ED models on the test set of LEVEN. Please check our paper for more details.

![image-20220306231007067](C:\Users\leoyao\AppData\Roaming\Typora\typora-user-images\image-20220306231007067.png)

## CodaLab

To get the test results, you can submit your predictions to our CodaLab competition (link is coming soon). 

## Codes

We release the source codes for the event detection baselines and downstream tasks.

​	The Baselines folder includes [DMCNN](./Baselines/DMCNN), [BiLSTM](./Baselines/BiLSTM), [BiLSTM+CRF](./Baselines/BiLSTM+CRF), [BERT](./Baselines/BERT), [BERT+CRF](./Baselines/BERT+CRF), [DMBERT](./Baselines/DMBERT).

​	The Downstreams folder includes [Legal Judgment Prediction](./Downstreams/LJP) and [Similar Case Retrieval](./Downstreams/SCR).

## Citation

If these data and codes help you, please cite this paper.

## Event Schema

The Chinese event schema is shown below. Please check our paper for the English version.![image-20220306231327699](C:\Users\leoyao\AppData\Roaming\Typora\typora-user-images\image-20220306231327699.png)