from copy import deepcopy
import itertools
import random
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from flair.data import Dictionary, Sentence
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from abc import abstractmethod
import pdb

class Result(object):
    def __init__(
        self, main_score: float, log_header: str, log_line: str, detailed_results: str, name: str = None,
    ):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results
        self.name = name


class Metric(object):
    def __init__(self, name):
        self.name = name

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)
            > 0
        ):
            return round(
                (self.get_tp(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                ),
                4,
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


class MetricRegression(object):
    def __init__(self, name):
        self.name = name

        self.true = []
        self.pred = []

    def mean_squared_error(self):
        return mean_squared_error(self.true, self.pred)

    def mean_absolute_error(self):
        return mean_absolute_error(self.true, self.pred)

    def pearsonr(self):
        return pearsonr(self.true, self.pred)[0]

    def spearmanr(self):
        return spearmanr(self.true, self.pred)[0]

    ## dummy return to fulfill trainer.train() needs
    def micro_avg_f_score(self):
        return self.mean_squared_error()

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_MEAN_SQUARED_ERROR\t{0}_MEAN_ABSOLUTE_ERROR\t{0}_PEARSON\t{0}_SPEARMAN".format(
                prefix
            )

        return "MEAN_SQUARED_ERROR\tMEAN_ABSOLUTE_ERROR\tPEARSON\tSPEARMAN"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        line = "mean squared error: {0:.4f} - mean absolute error: {1:.4f} - pearson: {2:.4f} - spearman: {3:.4f}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )
        return line


class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"


class WeightExtractor(object):
    def __init__(self, directory: Path, number_of_weights: int = 10):
        self.weights_file = init_output_file(directory, "weights.txt")
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            # if len(vec.size())==0:
            #     continue
            if len(vec.size())>0:
                weights_to_watch = min(
                    self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size()))
                )
            else:
                weights_to_watch=1
            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]

                for index in self.weights_dict[key][i]:
                    vec = vec[index]
                value = vec.item()

                with open(self.weights_file, "a") as f:
                    f.write("{}\t{}\t{}\t{}\n".format(iteration, key, i, float(value)))

    def _init_weights_index(self, key, state_dict, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = state_dict[key]
            cur_indices = []

            for x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        self.weights_dict[key] = indices


class Dejavuer:

    def __init__(self, cache=None):
        self.cache = set()
        if cache is not None:
            self.cache = cache

    def dejavu(self, data):
        text, *others = data
        text = normalize(text)
        data = (text, *others)
        if data in self.cache:
            return True
        else:
            self.cache.add(data)
            return False


def normalize(text):
	return ''.join(e for e in text.lower() if e.isalnum())


def init_output_file(base_path: Path, file_name: str, mode='w') -> Path:
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, mode, encoding="utf-8").close()
    return file


def convert_labels_to_one_hot(
    label_list: List[List[str]], label_dict: Dictionary
) -> List[List[int]]:
    """
    Convert list of labels (strings) to a one hot list.
    :param label_list: list of labels
    :param label_dict: label dictionary
    :return: converted label list
    """
    return [
        [1 if l in labels else 0 for l in label_dict.get_items()]
        for labels in label_list
    ]


def log_line(log):
    log.info("-" * 100)


def add_file_handler(log, output_file, mode='w'):
    init_output_file(output_file.parents[0], output_file.name, mode)
    fh = logging.FileHandler(output_file, mode=mode, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh


def store_embeddings(sentences: List[Sentence], storage_mode: str):

    # if memory mode option 'none' delete everything
    if storage_mode == "none":
        for sentence in sentences:
            sentence.clear_embeddings()

    # else delete only dynamic embeddings (otherwise autograd will keep everything in memory)
    else:
        # find out which ones are dynamic embeddings
        delete_keys = []
        for name, vector in sentences[0][0]._embeddings.items():
            if sentences[0][0]._embeddings[name].requires_grad:
                delete_keys.append(name)

        # find out which ones are dynamic embeddings
        for sentence in sentences:
            sentence.clear_embeddings(delete_keys)

    # memory management - option 1: send everything to CPU
    if storage_mode == "cpu":
        for sentence in sentences:
            sentence.to("cpu")


def store_teacher_predictions(sentences: List[Sentence], storage_mode: str):

    if storage_mode == "cpu":
        for sentence in sentences:
            sentence.store_teacher_prediction(storage_mode)


def get_all_metrics(data, tag_type, add_surface_form=False, eval_original=False):
    metrics = {}
    params_dicts = [
        {  # recovered
            'pred_tag_type': 'predicted',
            'orig_tag_type': 'predict',  # used for keeping original if no nonlocals
        },
    ]
    if add_surface_form:
        params_dicts.append({  # recovered (surface form)
            'pred_tag_type': 'predicted',
            'orig_tag_type': 'predict',
            'use_surface_form': True,
            'suffix': ' (Surface Form)'
        })
    if eval_original:
        params_dicts.append({  # original performance without recovering
            'pred_tag_type': 'predict',
            'suffix': ' [Original]',
        })
        if add_surface_form:
            params_dicts.append({  # original performance without recovering
                'pred_tag_type': 'predict',
                'use_surface_form': True,
                'suffix': ' [Original] (Surface Form)',
            })
        
    for params in params_dicts:
        _metrics = get_metrics(data, remove_x=True, tag_type=tag_type, **params)
        assert len(set(_metrics.keys()).intersection(set(metrics.keys()))) == 0
        metrics.update(_metrics)
    return metrics


def get_metrics(batch, remove_x, tag_type, pred_tag_type, orig_tag_type=None, use_surface_form=False, cache_for_dejavuer=None, suffix='') -> List[Metric]:
    """
    Args
        - orig_tag_type: original tag name used for keeping original if no nonlocals; do not keep anyhow if set to None
    """
    metrics = {
        "ner": "NER",
        "only_ex_w_nlc": "Only Examples with Nonlocals",
        "only_ex_wo_nlc": "Only Examples with Locals",
    }

    _metric_params = {
        "ner": {},
        "only_ex_w_nlc": {"example_filter": "only_ex_w_nonlocals"},
        "only_ex_wo_nlc": {"example_filter": "only_ex_wo_nonlocals"},
    }

    if orig_tag_type is not None:
        metrics["keep_orig_if_no_nlc"] = "Keep Original Predictions If No Nonlocals"
        _metric_params["keep_orig_if_no_nlc"] = {"keep_orig_if_no_nonlocals": True, "orig_tag_type": orig_tag_type}

    # suffix the key name
    if suffix != '':
        for name in list(metrics.keys()):
            new_name = name + suffix
            _metric_params[new_name] = _metric_params.pop(name)
            metrics[new_name] = metrics.pop(name) + suffix

    if use_surface_form:
        for name in metrics.keys():
            _metric_params[name].update({
                    'surface_form': True, 
                    'dejavuer': Dejavuer(cache_for_dejavuer),
                })
    
    for name in metrics.keys():
        metrics[name] = Metric(metrics[name])

    for name, metric in metrics.items():
        add_to_metric(batch, metric, remove_x, tag_type, pred_tag_type, **_metric_params[name])
    return metrics


def add_to_metric(
	batch,
	metric,
	remove_x,
	gold_tag_type,
	predict_tag_type,
	keep_orig_if_no_nonlocals=False,
	orig_tag_type=None,
	example_filter=None,
	surface_form=False,
	dejavuer: Dejavuer = None,
):
    for sentence in batch:
        
        if example_filter == 'only_ex_w_nonlocals':
            if not sentence.has_nonlocals:
                continue
        elif example_filter == 'only_ex_wo_nonlocals':
            if sentence.has_nonlocals:
                continue
        else:
            if example_filter is not None:
                raise ValueError(example_filter)
        
        # FIXED: (cwhsu) there might be a rare situation of an unexpected bug in CLNER's original code that an NE will be correct if there is another NE in the sentence that matches the predicted one.
        # solution: add position into the tuple
        # make list of gold tags
        gold_tags = [
            (tag.tag, str(tag)) for tag in sentence.get_spans(gold_tag_type)
        ]
        # make list of predicted tags
        _predict_tag_type = predict_tag_type
        if keep_orig_if_no_nonlocals:
            assert orig_tag_type is not None
            if not sentence.has_nonlocals:
                _predict_tag_type = orig_tag_type
        predicted_tags = [
            (tag.tag, str(tag)) for tag in sentence.get_spans(_predict_tag_type)
        ]

        if remove_x:

            # gold_tags_info = [[t.idx for t in tag.tokens] for tag in sentence.get_spans(gold_tag_type)]
            predicted_tags_info = [[t.idx for t in tag.tokens] for tag in sentence.get_spans(_predict_tag_type)]
            new_predicted_tags = []
            for tag_idx, tags in enumerate(predicted_tags):
                flag = 0
                # stride=ast.literal_eval(re.match('.*\-span (\[.*\])\:.*',tags[1]).group(1))
                stride = predicted_tags_info[tag_idx]
                for val in stride:
                    if sentence[val-1].get_tag(gold_tag_type).value == 'S-X':
                        flag = 1
                        break
                if not flag:
                    # new_gold_tags.append(tags)
                    new_predicted_tags.append(tags)
            predicted_tags = new_predicted_tags
            new_gold_tags = [x for x in gold_tags if x[0] != 'X']
            gold_tags = new_gold_tags

        # check for true positives, false positives and false negatives
        for tag, prediction in predicted_tags:
            if surface_form and dejavuer.dejavu((tag, prediction)):
                continue

            if (tag, prediction) in gold_tags:
                metric.add_tp(tag)
            else:
                metric.add_fp(tag)

        for tag, gold in gold_tags:
            if surface_form and dejavuer.dejavu((tag, gold)):
                continue

            if (tag, gold) not in predicted_tags:
                metric.add_fn(tag)
    return metric


def get_result_from_metric(metric: Metric):
	detailed_result = (
					f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
					f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
				)
	for class_name in metric.get_classes():
		detailed_result += (
							f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
							f"fn: {metric.get_fn(class_name)} - precision: "
							f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
							f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
							f"{metric.f_score(class_name):.4f}"
						)

	result = Result(
					main_score=metric.micro_avg_f_score(),
					log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
					log_header="PRECISION\tRECALL\tF1",
					detailed_results=detailed_result,
					name=metric.name,
				)
	
	return result


def log_result(log, result: Result, verbose=True):
    log_line(log)
    log.info(f"=== {result.name} ===")
    if hasattr(result, 'num_sents'):
        log.info(f"# examples: {result.num_sents}")
    log.info(result.log_line)
    if verbose:
        log.info(result.detailed_results)
    log_line(log)