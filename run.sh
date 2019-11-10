#!/bin/bash
export BERT_BASE_DIR=/Users/somdutta/bert/uncased_L-12_H-768_A-12
python /Users/somdutta/bert/run_classifier.py \
--task_name=cola \
--do_train=true \
--do_eval=true \
--data_dir=/Users/somdutta/bert/data/tsv \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--do_lower_case=True \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./bert_output/


