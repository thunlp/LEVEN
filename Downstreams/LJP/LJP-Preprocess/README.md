# Data Preprocess for Downstream Task1-LJP
This code is the data preprocessing for LEVEN downstream application LJP (Legal Judgment Prediction).

- Tokenize the original text and add the `input_ids` to the data files. 
  
    `input_ids` are the input for the BERT-based models.


- Detect events in the original text and add `token_type_ids` and `event_type_ids` to the data files.

    `token_type_ids` are binary indication of whether a token is trigger or not.

    `event_type_ids` indicate the specific event type id within the 108 event types.

## Requirements
- python==3.6.9

- torch==1.7.1

- transformers==4.12.5


## Usage
1. Download and unzip the [CAIL-2018](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) dataset files in 
   the `./input_data` folder.
   
2. Download the checkpoint from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/2cb9d439a7a547e0a21a/) or [Google Drive](https://drive.google.com/drive/folders/1bkd08NIGHd1ZG_lioSP3z6-0RBLo6vsY?usp=sharing) and put the folder to `./saved`.
3. Run `python process.py` and the processed data will be generated in `./output_data/small` folder.
   (We used the `CAIL2018-small` in our paper)
   
4. The data we used for the low-resource LJP can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6e2342a1d8754c569b64/) 
   or [Google Drive](https://drive.google.com/drive/folders/1507RbKauVLyNORldqpag-FkmE_3DOAxo?usp=sharing).
   
    There are 5 files in the link since we run the experiment 5 times and report the average performance.
   
5. Put the processed files above to the [LJP-Experiment](../LJP-Experiment) folder.

    - Put `small` folder in step 3 into `/LJP-Experiment/input_data/`
    - Put `fewshot` folder in step 4 into `/LJP-Experiment/`

    
## Hint
Run this code on a GPU machine can speed up the whole process. 

We ran the code on RTX-2080TI and it took around 50, 5, and 10 minutes to process `train.json`, `valid.json` and`test.json`, respectively.