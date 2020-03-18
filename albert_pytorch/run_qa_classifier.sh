export BERT_BASE_DIR=/home/speech/models/albert_tiny_pytorch_489k
TASK_NAME="qa"

python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=/home/speech/data/single_qa \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --max_seq_length=64 \
  --per_gpu_train_batch_size=128 \
  --per_gpu_eval_batch_size=128 \
  --learning_rate=1e-4 \
  --num_train_epochs=5.0 \
  --logging_steps=26455 \
  --save_steps=26455 \
  --output_dir=/home/speech/models/qa_output_ \
  --overwrite_output_dir