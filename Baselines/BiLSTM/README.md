# BiLSTM
The code is the implementation of BiLSTM for event detection on LEVEN. 

## Requirements

+ torch==1.6
+ numpy
+ sklearn
+ seqeval==1.2.2
+ tqdm==4.44.0

## Usage

To run this code, you need to:
1. put raw files of LEVEN dataset and [pretrained word embeddings](https://github.com/Embedding/Chinese-Word-Vectors) in `./raw`

    ( We use the embeddings trained by Wikipedia_zh with Word+Character+Ngram context features.)
2. run ```python main.py --config [path of config files] --gpu [gpu, optional]```  
we will train, evaluate and test models in every epoch. We output the performance of training and evaluating, and generate test result files for submit to CodaLab (link is coming soon).

All the hyper-parameters are in config file at `./config/`, you can modify it as you wish.
