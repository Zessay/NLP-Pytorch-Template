python process_multiQA_pipline.py \
  --from_path=/home/speech/data/es_data \
  --from_file=personality.data_0303.json \
  --to_path=/home/speech/data/multi_0326 \
  --to_file_name=persona \
  --type=uur \
  --sample=random \
  --train_rate=0.8 \
  --seed=42 \
  --n_jobs=1