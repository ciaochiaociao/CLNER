#!/bin/bash

function get_data() {
	LOCAL="$1"
	SENT="$2"
	file="$3"
	sed 's/ /\tO\n/g' <(echo -n "${LOCAL}") > $file
	echo -e '<EOS> B-X' >> $file
	echo "$(sed 's/ / B-X\n/g' <(echo "${SENT}"))" >> $file

	cat $file

	echo "The above content is outputted to $file"
}


#conda activate CLNER

#dataset_name=wnut17
#dataset_name=globalner@N-google@Q-sent@S-title_and_snippet@R-bertscore
#dataset_name=globalner@N-M-google-S-google@Q-mention_and_sent@S-title_and_snippet@R-bertscore
#dataset_name=globalner@N-M-google-S-google@Q-mention_and_sent@S-title_and_snippet@R-mentionscore+bertscore  # the one mostly used in error_analysis.xlsm
dataset_name=globalner@N-M-google-S-google@Q-mention_and_sent@S-title_and_snippet@R-mentionscore_soft+bertscore

#run=4
#run=7
#run=1
#run=8
run=4

train_config_file=config/$dataset_name$run.yaml

model_path_suffix=$dataset_name$run
# model_path=/home/cwhsu/.models/xlmr-first_20epoch_2batch_2accumulate_0.000005lr_10000lrrate_eng_monolingual_crf_fast_norelearn_sentbatch_sentloss_finetune_nodev_$model_path_suffix
model_path=/home/hlv8980/recovery_on_172/resources/taggers/'A21->A22@ep-5->A23'
dataset_path=tmp_dataset
output_result_path=tmp_eval_results

config_file=$output_result_path/eval_config.yaml

get_data "$1" "$2" $dataset_path/dev.txt

mkdir -p $output_result_path

set -e
python generate_config_eval.py \
  $model_path \
  $dataset_path \
  $train_config_file \
  $output_result_path \
  $config_file 
set +e
echo '============'
echo local: $LOCAL >> infer.log
echo sent: $SENT >> infer.log
echo model_path: $model_path >> infer.log

comm -1 -3 --nocheck-order config/template.yaml $config_file | tee $log_file | tee -a infer.log

set -e
python train.py --config $config_file --test > $output_result_path/eval.log
cat $output_result_path/dev.tsv | tee -a infer.log
echo 'Success!' >> infer.log
set +e
