# LEVEN Downstream Task2 - SCR
This code is for LEVEN downstream task SCR. The implementation is based on [pytorch-worker](https://github.com/haoxizhong/pytorch-worker).

## Requirements
- python==3.6.9

- torch==1.7.1

- transformers==4.12.5

## Usage
1. You have to preprocess the data according to [SCR-Preprocess](../SCR-Preprocess) first.
   
2. For the supervised experiment, the detailed configuration is in `./config/pairwise/LecardBert.config`.
   
   - Run `run_train.sh` to train the model and generate the predicted files.
   - Run `python ./utils/gen_metric.py` to evaluate. See the `.py` file for details
   
3. For the unsupervised experiment, we compare our event-based method with this [repo](https://github.com/myx666/LeCaRD/#experiment).

   - run `python run_bag_of_event_w.py` to get the result of bag_of_event_w.
   - comment line 22 of `run_bag_of_event_w.py` and run it to get the result of bag_of_event.

## Hint

​	Please read and delete the comments including `space` and `#` in the `.config` file before running `run_train.sh`.