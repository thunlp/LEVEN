# DMBERT
This code is the implementation for [DMBERT](https://www.aclweb.org/anthology/N19-1105/) model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers), especially its example for the multiple-choice task.



## Requirements

- python==3.6.9

- torch==1.2.0

- transformers==2.8.0

- sklearn==0.20.2

  

## Usage

Hint: please read and delete all the comments after the backslash in each line of the ```.sh``` scripts before running them.

1. Download LEVEN data files and put them in the `./data` folder.
2. Run ```run_train.sh``` for training and evaluation on the validation set.  
3. Run ```run_infer.sh``` to get predictions on the test set (dumped to ```results.jsonl```).

See the two scripts for more details.

