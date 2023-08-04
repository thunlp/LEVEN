# Data Preprocess for Downstream Task2-SCR
This code is the data preprocessing for LEVEN downstream application SCR (Similar Case Retrieval).

- Tokenize the original text and add the `input_ids` to the data files. 
  
    `input_ids` are the input for the BERT-based models.


- Detect events in the original text and add `event_type_ids`/`event_ids` to the data files.

    `event_type_ids`/`event_ids` indicate the specific event type id within the 108 event types.


## Requirements
- python==3.6.9

- torch==1.7.1

- transformers==4.12.5


## Usage
1. Download and put the original [LeCaRD](https://github.com/myx666/LeCaRD/tree/main/data) dataset 
   files in the `./input_data` folder. (Only `candidates`, `label` , `prediction` and `query` folders are necessary.)
   
2. Download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/1bkd08NIGHd1ZG_lioSP3z6-0RBLo6vsY?usp=sharing) and put the folder to `./saved`.
   
3. Process data for both supervised and unsupervised settings.
    - run `python process4supervised.py`, it takes around 10 minutes on a GPU-machine.
    - run `python process4unsupervised.py`, it takes around 50 minutes on a GPU-machine.
    
4. For the supervised experiment, we adopt 5-fold validation. The data can be obtained from [Google Drive](https://drive.google.com/drive/folders/1yQZZ6Vs8kjFiWjwZCYmwGK8zghUdGYsJ?usp=sharing). Download the data and put it into `./output_data-supervised/` folder.
   
    It should be like `./output_data-supervised/query/query_5fold`.

5. Put the processed files above to the [SCR-Experiment](../SCR-Experiment) folder.
    - put `./input_data/label`, `./output_data-supervised/candidates` and `./output_data-supervised/query` together into 
   `./SCR-Experiment/input_data` folder for supervised experiment.
    - put `./output_data-unsupervised/candidates` and `./output_data-unsupervised/query` together into `./SCR-Experiment/input_data_unsupervised`.


## Hint
The data processing for the supervised setting and unsupervised setting are slightly different.

For the supervised setting, we process the data to feed into the BERT model which has a limited input length of 512. Therefore, we do prediction at document level and do truncation.

For the unsupervised setting, we process the data regardless of the length. Therefore, we use sentence-level predictions. And this causes the different consuming time.
