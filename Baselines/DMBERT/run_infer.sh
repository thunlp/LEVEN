python run_ee.py \
    --data_dir ./data/ \#path to the test data, remember to delete the cached files at first (otherwise the test data may be random shuffled before)
    --model_type bert \
    --model_name_or_path ./saved/checkpoint-500 \#path to the trained checkpoint
    --task_name leven_infer \
    --output_dir ./saved \#output path
    --max_seq_length 512 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_infer #add this flag to do inference only

python get_submission.py \#convert the predictions to the submission format
    --test_data ./data/test.jsonl \#path to the test data file
    --preds ./saved/checkpoint-500/checkpoint-500_preds.npy \#path to the prediction file
    --output ./saved/checkpoint-500/results.jsonl #output file
