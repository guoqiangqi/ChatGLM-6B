
LR=1e-4

deepspeed --num_gpus=1 pre_train.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /root/data/mlm/train.json \
    --validation_file /root/data/mlm/train.json \
    --max_seq_length 128 \
    --overwrite_cache \
    --model_name_or_path /root/LLM/chatglm-6b \
    --output_dir ./output/chatglm-6b-pretrain-$LR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR 
