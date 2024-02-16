from time import ctime, time
from typing import List
import flair
from flair.data import Dictionary, Sentence, Token, Label
#from flair.datasets import CONLL_03, CONLL_03_DUTCH, CONLL_03_SPANISH, CONLL_03_GERMAN
import flair.datasets as datasets
from flair.data import MultiCorpus, Corpus
from flair.list_data import ListCorpus
import flair.embeddings as Embeddings
from flair.training_utils import EvaluationMetric, add_file_handler, get_all_metrics, get_result_from_metric, log_result
from flair.visual.training_curves import Plotter
# initialize sequence tagger
# from flair.models import SequenceTagger
from pathlib import Path
import argparse
import yaml
from flair.utils.from_params import Params
# from flair.trainers import ModelTrainer
# from flair.trainers import ModelDistiller
# from flair.trainers import ModelFinetuner
from flair.config_parser import ConfigParser
import pdb
import sys
import os, shutil
import logging
from flair.custom_data_loader import ColumnDataLoader
from flair.datasets import ColumnDataset, DataLoader
from process import wait_for_process
# Disable
def blockPrint():
		sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
		sys.stdout = sys.__stdout__

parser = argparse.ArgumentParser('train.py')
parser.add_argument('--config', help='configuration YAML file.')
parser.add_argument('--test', action='store_true', help='Whether testing the pretrained model.')
parser.add_argument('--zeroshot', action='store_true', help='testing with zeroshot corpus.')
parser.add_argument('--all', action='store_true', help='training/testing with all corpus.')
parser.add_argument('--other', action='store_true', help='training/testing with other corpus.')
parser.add_argument('--quiet', action='store_true', help='print results only')
parser.add_argument('--nocrf', action='store_true', help='without CRF')
parser.add_argument('--parse', action='store_true', help='parse files')
parser.add_argument('--parse_train_and_dev', action='store_true', help='chech the performance on the training and development sets')
parser.add_argument('--keep_order', action='store_true', help='keep the parse order for the prediction')
parser.add_argument('--predict', action='store_true', help='predict files')
parser.add_argument('--debug', action='store_true', help='debugging')
parser.add_argument('--target_dir', default='', help='file dir to parse')
parser.add_argument('--spliter', default='\t', help='file dir to parse')
parser.add_argument('--recur_parse', action='store_true', help='recursively parse the file dirs in target_dir')
parser.add_argument('--parse_test', action='store_true', help='parse the test set')
parser.add_argument('--save_embedding', action='store_true', help='save the pretrained embeddings')
parser.add_argument('--mst', action='store_true', help='use mst to parse the result')
parser.add_argument('--test_speed', action='store_true', help='test the running speed')
parser.add_argument('--predict_posterior', action='store_true', help='predict the posterior distribution of CRF model')
parser.add_argument('--batch_size', default=-1, help='manually setting the mini batch size for testing')
parser.add_argument('--keep_embedding', default=-1, help='mask out all embeddings except the index, for analysis')

# by cwhsu
parser.add_argument('--sample', action='store_true')
parser.add_argument('--sample_ratio', type=float, default=0.1)
parser.add_argument('--pid_to_wait', type=int)
parser.add_argument('--force', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--inference_verbose', '-v', action='store_true')
parser.add_argument('--interactive', '-i', action='store_true')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--test_on_subsets', default='train,dev,test', help='train,dev,test (by default)')
parser.add_argument('--all_tag_prob', action='store_true')

def count_parameters(model):
	import numpy as np
	total_param = 0
	for name,param in model.named_parameters():
		num_param = np.prod(param.size())
		# print(name,num_param)
		total_param+=num_param
	return total_param


log = logging.getLogger("flair")
args = parser.parse_args()

wait_for_process(args.pid_to_wait)

if args.quiet:
	blockPrint()
	log.disabled=True
config = Params.from_file(args.config)

if args.test:
	log_handler = add_file_handler(log, Path(config['target_dir'] + "/" + config['model_name'] + "/testing.log"))

if args.test and args.zeroshot:
	temperory_reject_list=['ast','enhancedud','dependency','atis','chunk']
	if config['targets'] in temperory_reject_list:
		enablePrint()
		print()
		exit()

# pdb.set_trace()
config = ConfigParser(config,all=args.all,zero_shot=args.zeroshot,other_shot=args.other,predict=args.predict,inference=args.inference,load_corpus_from_target_path=args.only_eval)
os.makedirs(config.get_target_path, exist_ok=args.force)
try:
	shutil.copy(args.config, config.get_target_path)
except shutil.SameFileError:
    pass
# pdb.set_trace()

from pprint import pprint
pprint(args)

# import pdb; pdb.set_trace()
# toy corpus for testing by cwhsu
# if args.toy_test:
# 	config.corpus=config.corpus.downsample(0.05)

corpus=config.corpus

if args.sample:
	log.info(f'Before sampling => {str(corpus)}')
	corpus.downsample(args.sample_ratio)
	log.info(f'After sampling with ratio {args.sample_ratio} => {str(corpus)}')

if args.only_eval:
	log_handler = add_file_handler(log, config.get_target_path / "eval.log")
	for dataset in ('train', 'dev', 'test'):
		log.info(f"===== {dataset} =====")
		dataset = getattr(corpus, dataset)
		metrics = get_all_metrics(dataset, config.get_target, add_surface_form=True, eval_original=True)
		results = [get_result_from_metric(metric) for metric in metrics.values()]
		for result in results:
			log_result(log, result)
	log.removeHandler(log_handler)
	exit()

if args.inference:
    if config.config.get('load_pretrained', False):
        config.config['load_pretrained'] = False
    if 'pretrained_model' in config.config:
    	del config.config['pretrained_model']

student=config.create_student(nocrf=args.nocrf)
print(student)
log.info(f"Model Size: {count_parameters(student)}")

teacher_func=config.create_teachers
if 'is_teacher_list' in config.config:
	if config.config['is_teacher_list']:
		teacher_func=config.create_teachers_list

# pdb.set_trace()
if 'trainer' in config.config:
	trainer_name=config.config['trainer']
else:
	if 'ModelDistiller' in config.config:
		trainer_name='ModelDistiller'
	elif 'ModelFinetuner' in config.config:
		trainer_name='ModelFinetuner'
	elif 'ReinforcementTrainer' in config.config:
		trainer_name='ReinforcementTrainer'
	else:
		trainer_name='ModelDistiller'

trainer_func=getattr(flair.trainers,trainer_name)


if 'distill_mode' not in config.config[trainer_name]:
	config.config[trainer_name]['distill_mode']=False
if not args.test and config.config[trainer_name]['distill_mode']:
	teachers=teacher_func()
	professors=[]
	# corpus=config.distill_teachers_prediction()
	trainer: trainer_func = trainer_func(student, teachers, corpus, config=config.config, professors=professors,**config.config[trainer_name])
elif not args.parse:
	trainer: trainer_func = trainer_func(student, None, corpus, config=config.config, **config.config[trainer_name], is_test=args.test)
else:
	trainer: trainer_func = trainer_func(student, None, corpus, config=config.config, **config.config[trainer_name], is_test=args.test)

# pdb.set_trace()

train_config=config.config['train']
train_config['base_path']=config.get_target_path

# train_config['shuffle']=False
eval_mini_batch_size = int(config.config['train']['mini_batch_size'])
# if args.parse or args.test:
#   if 'sentence_level_batch' in config.config[trainer_name] and config.config[trainer_name]['sentence_level_batch']:
#       eval_mini_batch_size = 2000
# pdb.set_trace()
if int(args.batch_size)>0:
	eval_mini_batch_size = int(args.batch_size)

if args.test_speed:
	student.eval()
	# pdb.set_trace()
	print(count_parameters(student))
	# for embedding in student.embeddings.embeddings:
	# 	embedding.training = False
	test_loader=ColumnDataLoader(list(trainer.corpus.test),32,use_bert=trainer.use_bert,tokenizer=trainer.bert_tokenizer, sort_data=False, model = student, sentence_level_batch = True)
	test_loader.assign_tags(student.tag_type,student.tag_dictionary)
	train_eval_result, train_loss = student.evaluate(test_loader,embeddings_storage_mode='none',speed_test=True)
	# print('Current accuracy: ' + str(train_eval_result.main_score*100))
	# print(train_eval_result.detailed_results)

elif args.inference:
	from flair.custom_data_loader import ColumnDataLoader
	import torch
	__model_path = config.get_target_path / "best-model.pt"
	logging.info(f"loading the model file from {str(__model_path)} for doing inference (--inference)")
	student = student.load(__model_path)
	student.eval()
	print(f"{student}\n")
	__corpus = corpus
	infer_cache_path = config.get_target_path / 'inference' / 'cache.log'

	infer_cache_fh = add_file_handler(log, infer_cache_path, mode='a')
	while True:
		try:
			__dataset = __corpus.test
			loader=ColumnDataLoader(list(__dataset), 2, use_bert=True, model = student, sentence_level_batch = True)
			loader.assign_tags(student.tag_type, student.tag_dictionary)
			with torch.no_grad():
				trainer.gpu_friendly_assign_embedding([loader])
			out_path = config.get_target_path / "inference" / "output.tsv"
			test_results, test_loss = student.evaluate(
				loader,
				out_path=out_path,
				embeddings_storage_mode="cpu",
			)
			if args.inference_verbose:
				for sent in __dataset:
					sent : Sentence = sent
					log.info('datetime:' + str(ctime(time())))
					log.info('command:' + repr(sys.argv))

					log.info('== gold ==')
					log.info(sent.to_tagged_string('ner'))
					log.info('== original ==')
					log.info(sent.to_tagged_string('predict'))
					log.info('== recovered ==')
					log.info(sent.to_tagged_string('predicted'))
				
				log.info('== token-by-token ==')
				with open(out_path) as f:
					lines = f.read()
					log.info('\n' + lines)

			log.info('== evaluation ==')
			# ------ 2023/10/24
			metrics = get_all_metrics(__dataset, config.get_target, add_surface_form=False, eval_original=False)
			results = [get_result_from_metric(metric) for metric in metrics.values()]
			for result in results:
				log_result(log, result)
			# for result_name, result in test_results.items():
			# 	log.info(f'=== {result_name} ===')
			# 	log.info(result.log_line)
			# 	log.info(result.detailed_results)
			# ------

			if not args.interactive:
				break
			
			# interactive modes
			_embed = True
			_cont = True
			while _embed:
				res = input(f"Continue (c) / Interactive Shell (i) / pdb (d) ? (Input New Data in '{config.get_target_path / 'inference' / 'input.tsv'}') ")
				if res[0].lower() == 'i':
					import IPython
					IPython.embed()
				elif res[0].lower() == 'd':
					pdb.set_trace()
				else:
					_embed = False
					if res[0].lower() != 'c':
						_cont = False
					break  # exit embed mode
			if not _cont:
				break  # exit interactive inference mode

			__corpus = config.get_inference_corpus
		except Exception as e:
			print(repr(e))
			if input(f"An error occurred as shown above !! Continue ? (y/n) ") != 'y':
				raise e
	log.removeHandler(infer_cache_fh)

elif args.test:
	# import pdb; pdb.set_trace()
	student.eval()
	trainer.embeddings_storage_mode = 'cpu'
	trainer.final_test(
		config.get_target_path,
		eval_mini_batch_size=eval_mini_batch_size,
		overall_test=True if int(args.keep_embedding)<0 else False,
		quiet_mode=args.quiet,
		nocrf=args.nocrf,
		# debug=args.debug,
		# keep_embedding = int(args.keep_embedding),
		predict_posterior=args.predict_posterior,
		# sort_data = not args.keep_order,
		out_pathspec=str(config.get_target_path / 'infer_{}.tsv'),
		subsets=tuple(args.test_on_subsets.strip().split(',')),
  		all_tag_prob=args.all_tag_prob,
	)
	log.removeHandler(log_handler)
elif args.parse or args.save_embedding:
	print('Batch Size:',eval_mini_batch_size)
	base_path=Path(config.config['target_dir'])/config.config['model_name']
	if (base_path / "best-model.pt").exists():
		print('Loading pretraining best model')
		if trainer_name == 'ReinforcementTrainer':
			student = student.load(base_path / "best-model.pt", device='cpu')
			for name, module in student.named_modules():
				if 'embeddings' in name or name == '':
					continue
				else:
					module.to(flair.device)
			for name, module in student.named_parameters():
				module.to(flair.device)
		else:
			student = student.load(base_path / "best-model.pt")
		
	elif (base_path / "final-model.pt").exists():
		print('Loading pretraining final model')
		student = student.load(base_path / "final-model.pt")
	else:
		assert 0, str(base_path)+ ' not exist!'
	if trainer_name == 'ReinforcementTrainer':
		import torch
		training_state = torch.load(base_path/'training_state.pt')
		start_episode = training_state['episode']
		student.selection = training_state['best_action']
		name_list=sorted([x.name for x in student.embeddings.embeddings])
		print(name_list)
		print(f"Setting embedding mask to the best action: {student.selection}")
		embedlist = sorted([(embedding.name, embedding) for embedding in student.embeddings.embeddings], key = lambda x: x[0])
		for idx, embedding_tuple in enumerate(embedlist):
			embedding = embedding_tuple[1]
			if student.selection[idx] == 1:
				embedding.to(flair.device)
				if 'elmo' in embedding.name:
					# embedding.reset_elmo()
					# continue
					# pdb.set_trace()
					embedding.ee.elmo_bilm.cuda(device=embedding.ee.cuda_device)
					states=[x.to(flair.device) for x in embedding.ee.elmo_bilm._elmo_lstm._states]
					embedding.ee.elmo_bilm._elmo_lstm._states = states
					for idx in range(len(embedding.ee.elmo_bilm._elmo_lstm._states)):
						embedding.ee.elmo_bilm._elmo_lstm._states[idx]=embedding.ee.elmo_bilm._elmo_lstm._states[idx].to(flair.device)
			else:
				embedding.to('cpu')
				
		for name, module in student.named_modules():
			if 'embeddings' in name or name == '':
				continue
			else:
				module.to(flair.device)
		parameters = [x for x in student.named_parameters()]
		for parameter in parameters:
			name = parameter[0]
			module = parameter[1]
			module.data.to(flair.device)
			if '.' not in name:
				if type(getattr(student, name))==torch.nn.parameter.Parameter:
					setattr(student, name, torch.nn.parameter.Parameter(getattr(student,name).to(flair.device)))
		# pdb.set_trace()
		
	if args.save_embedding:
		for embedding in student.embeddings.embeddings:
			if hasattr(embedding,'fine_tune') and embedding.fine_tune: 
				if not os.path.exists(base_path/embedding.name.split('/')[-1]):
					os.mkdir(base_path/embedding.name.split('/')[-1])
				embedding.tokenizer.save_pretrained(base_path/embedding.name.split('/')[-1])
				embedding.model.save_pretrained(base_path/embedding.name.split('/')[-1])
		exit()
	if not hasattr(student,'use_bert'):
		student.use_bert=False
	if hasattr(student,'word_map'):
		word_map = student.word_map
	else:
		word_map = None
	if hasattr(student,'char_map'):
		char_map = student.char_map
	else:
		char_map = None
	if args.mst:
		student.is_mst=True
	if args.parse_train_and_dev:

		print('Current Model: ', config.config['model_name'])
		print('Current Set: ', 'dev')
		if not os.path.exists('system_pred'):
			os.mkdir('system_pred')
		for index, subcorpus in enumerate(corpus.dev_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(loader,embeddings_storage_mode='none',
				out_path=Path('system_pred/dev.'+config.config['model_name']+'.conllu'),)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
		print('Current Set: ', 'train')
		for index, subcorpus in enumerate(corpus.train_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(
				loader,
				embeddings_storage_mode='none',
				out_path=Path('system_pred/train.'+config.config['model_name']+'.conllu'),
			)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
		# print('Current Set: ', 'train+dev')
		# for index, subcorpus in enumerate(corpus.train_list):
		# 	# log_line(log)
		# 	# log.info('current corpus: '+self.corpus.targets[index])
		# 	print('Current Lang: ', corpus.targets[index])
		# 	loader=ColumnDataLoader(list(subcorpus)+list(corpus.dev_list[index]),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order)
		# 	loader.assign_tags(student.tag_type,student.tag_dictionary)
		# 	train_eval_result, train_loss = student.evaluate(
		# 		loader,
		# 		embeddings_storage_mode='none',
		# 		out_path=Path('outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'),
		# 	)
		# 	print('Current accuracy: ' + str(train_eval_result.main_score*100))
		# 	print(train_eval_result.detailed_results)
		print('Current Set: ', 'test')
		for index, subcorpus in enumerate(corpus.test_list):
			# log_line(log)
			# log.info('current corpus: '+self.corpus.targets[index])
			if len(subcorpus)==0:
				continue
			print('Current Lang: ', corpus.targets[index])
			loader=ColumnDataLoader(list(subcorpus),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(
				loader,
				embeddings_storage_mode='none',
				out_path=Path('system_pred/test.'+config.config['model_name']+'.conllu'),
			)
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
	elif args.target_dir != '':
		if args.recur_parse:
			file_dirs=os.listdir(args.target_dir)
			for file_dir in file_dirs:
				tar_dir=os.path.join(args.target_dir,file_dir)
				if not os.path.isdir(tar_dir):
					continue
				if student.tag_type=='dependency':
					corpus=datasets.UniversalDependenciesCorpus(tar_dir,add_root=True,spliter=args.spliter)
				else:
					corpus=datasets.ColumnCorpus(tar_dir, column_format={0: 'text', 1:'ner'}, tag_to_bioes='ner')
				tar_file_name = tar_dir.split('/')[-1]
				print('Parsing the file: '+tar_file_name)
				write_name='outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'
				print('Writing to file: '+write_name)
				loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
				loader.assign_tags(student.tag_type,student.tag_dictionary)
				train_eval_result, train_loss = student.evaluate(loader,out_path=Path(write_name),embeddings_storage_mode="none",prediction_mode=True)
				if train_eval_result is not None:
					print('Current accuracy: ' + str(train_eval_result.main_score*100))
					print(train_eval_result.detailed_results)
		else:
			if student.tag_type=='dependency' or student.tag_type=='enhancedud':
				corpus=datasets.UniversalDependenciesCorpus(args.target_dir,add_root=True,spliter=args.spliter)
			else:
				corpus=datasets.ColumnCorpus(args.target_dir, column_format={0: 'text', 1:'ner'}, tag_to_bioes='ner')
			tar_file_name = str(Path(args.target_dir)).split('/')[-1]
			loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
			loader.assign_tags(student.tag_type,student.tag_dictionary)
			train_eval_result, train_loss = student.evaluate(loader,out_path=Path('outputs/train.'+config.config['model_name']+'.'+tar_file_name+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
			if train_eval_result is not None:
				print('Current accuracy: ' + str(train_eval_result.main_score*100))
				print(train_eval_result.detailed_results)
	elif args.parse_test:
		loader=ColumnDataLoader(list(corpus.test),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
		loader.assign_tags(student.tag_type,student.tag_dictionary)
		train_eval_result, train_loss = student.evaluate(loader,out_path=Path('system_pred/test.'+config.config['model_name']+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
		if train_eval_result is not None:
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
	else:
		loader=ColumnDataLoader(list(corpus.train),eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not args.keep_order, sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
		loader.assign_tags(student.tag_type,student.tag_dictionary)
		train_eval_result, train_loss = student.evaluate(loader,out_path=Path('outputs/train.'+config.config['model_name']+'.'+corpus.targets[0]+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
		if train_eval_result is not None:
			print('Current accuracy: ' + str(train_eval_result.main_score*100))
			print(train_eval_result.detailed_results)
else:
	getattr(trainer,'train')(**train_config)
# trainer.train(
#   config.get_target_path,
#   learning_rate=0.1,
#   mini_batch_size=32,
#   max_epochs=150
# )

