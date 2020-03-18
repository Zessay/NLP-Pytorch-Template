export BERT_BASE_DIR=/home/speech/models/albert_tiny_pytorch_489k
TASK_NAME="style"

python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=/home/speech/data/style_data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --max_seq_length=64 \
  --per_gpu_train_batch_size=64 \
  --per_gpu_eval_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_epochs=5.0 \
  --logging_steps=674 \
  --save_steps=674 \
  --output_dir=/home/speech/models/style_output_ \
  --overwrite_output_dir \
  --save_eval