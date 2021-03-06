python run_leven.py \
    --model_type bertcrf \
    --output_dir ./saved/checkpoint-100 \#path to the trained checkpoint, the results file will also be dumped here
    --max_seq_length 512 \
    --do_lower_case \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --save_steps 100 \
    --logging_steps 100 \
    --seed 0 \
    --do_infer #add this flag to do inference only
