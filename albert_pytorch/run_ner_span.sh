CURRENT_DIR=`pwd`
export BERT_BASE_DIR=/home/speech/models/albert_tiny_pytorch_489k
TASK_NAME="cluener"

python run_ner_span.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=/home/speech/data/cluener/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --logging_steps=672 \
  --save_steps=672 \
  --output_dir=/home/speech/models/${TASK_NAME}_output_/ \
  --overwrite_output_dir \
  --seed=42
