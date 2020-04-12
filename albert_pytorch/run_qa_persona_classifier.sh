export BERT_BASE_DIR=/home/speech/models/qa_output_albert/qa_model7675
TASK_NAME="qa"

python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=/home/speech/data/qa_persona \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --max_seq_length=64 \
  --per_gpu_train_batch_size=64 \
  --per_gpu_eval_batch_size=128 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --logging_steps=317 \
  --save_steps=317 \
  --output_dir=/home/speech/models/qa_persona_ \
  --loss_type=ls \
  --ls_epsilon=0.1 \
  --hier_lr \
  --overwrite_output_dir