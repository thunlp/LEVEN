# DMCNN
The code is the implementations of [DMCNN](https://www.aclweb.org/anthology/P15-1017/) for event detection on LEVEN. 

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
we will train, evaluate and test models in every epoch. We output the performance of training and evaluating, and generate test result files for submit to CodaLab  (link is coming soon).

All the hyper-parameters are in config files at `./config/`, you can modify them as you wish.
