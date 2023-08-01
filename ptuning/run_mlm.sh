python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --output_dir /root/ChatGLM-6B/ptuning/test-mlm \
    --use_auth_token True

# python run_mlm.py \
#     --model_name_or_path roberta-base \
#     --train_file /root/data/mlm/train.json \
#     --validation_file /root/data/mlm/train.json \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --do_train \
#     --output_dir /root/ChatGLM-6B/ptuning/test-mlm