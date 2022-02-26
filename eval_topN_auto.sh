#!/bin/bash

set -e
trap 'echo "Failed on line $LINENO, run $RANKER top$N :("; echo "$0 failed :(" | ssh 140.109.19.51 "cat | mail cwhsu.linux@gmail.com"' ERR
for N in {7..12}; do
	for RANKER in mentionscore+bertscore mentionscore_soft+bertscore bertscore bertscore_rec; do
		echo "Start running $RANKER top$N ..."
		sed "s/\$N/$N/g" < config/eval_topN_auto.yaml | sed "s/\$RANKER/$RANKER/g" > /tmp/config.yaml
		model_name=$(grep -oP '(?<=model_name: ).*' /tmp/config.yaml)
		target_dir=$(grep -oP '(?<=target_dir: ).*' /tmp/config.yaml)
		echo "model_name: $model_name"
		echo "target_dir: $target_dir"
		output_dir="$target_dir/$model_name"
		mkdir "$output_dir"
		python train.py --config /tmp/config.yaml --test > "$output_dir/eval.log"
		echo "Finished run $RANKER top$N"
	done
done
echo "Finished successfully :)"
echo "Finished successfully $0" | ssh 140.109.19.51 "cat | mail cwhsu.linux@gmail.com"
set +e
