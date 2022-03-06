# LEVEN Downstream Task1 - LJP
This code is for LEVEN downstream task LJP. The implementation is based on [CLAIM](https://github.com/thunlp/CLAIM/tree/ljp).

## Requirements
- python==3.6.9

- torch==1.7.1

- transformers==4.12.5

## Usage
1. You have to preprocess the data according to [LJP-Preprocess](../LJP-Preprocess) first.
   
2. In the directory ``config/ljp/small``, there exists two different configurations:

    - `bert.config`: full-data setting
    - `bert4fewshot.config`: low-resource setting
    
    Toggle `use_event_type` variable in the `.config` files to use event or not.
    
3. Refer to `run_train.sh` for usage details.
