# LEVEN
Dataset and source code for ACL 2022 Findings paper "LEVEN: A Large-Scale Chinese Legal Event Detection Dataset".

## Data

The dataset (ver. 1.0) can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6e911ff1286d47db8016/) or [Google Drive](https://drive.google.com/drive/folders/1VGD0h365kegTqGEyLr24SJtJUUoZIt20?usp=sharing). The data format is introduced in [this document](DataFormat.md).

## CodaLab

To get the test results, you can submit your predictions to our CodaLab competition (link is coming soon). 

## Codes

We release the source codes for the event detection baselines and downstream tasks.

​	The Baselines folder includes [DMCNN](./Baselines/DMCNN), [BiLSTM](./Baselines/BiLSTM), [BiLSTM+CRF](./Baselines/BiLSTM+CRF), [BERT](./Baselines/BERT), [BERT+CRF](./Baselines/BERT+CRF), [DMBERT](./Baselines/DMBERT).

​	The Downstreams folder includes [Legal Judgment Prediction](./Downstreams/LJP) and [Similar Case Retrieval](./Downstreams/SCR).

## Citation

If these data and codes help you, please cite this paper.
