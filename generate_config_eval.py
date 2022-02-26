import yaml
import sys
import os
from yaml import Loader

pretrained_model_path = sys.argv[1]
dataset_path = sys.argv[2]
train_config_path = sys.argv[3]
output_result_path = sys.argv[4]
output_config_path = sys.argv[5]

with open(train_config_path) as f:
  config = yaml.load(f, Loader=Loader)

output_result_parent = os.path.dirname(output_result_path)
output_result_folder_name = os.path.basename(output_result_path)

config['ner']['ColumnCorpus-WNUTDOCFULL']['data_folder'] = dataset_path
config['target_dir'] = output_result_parent
config['model_name'] = output_result_folder_name
config['load_pretrained'] = True
config['pretrained_model'] = pretrained_model_path

if os.path.exists(output_config_path):
  if os.environ.get('FORCE') != 'true' and input('Replace existing file from ' + output_config_path + ' ? (y/n)') != 'y':
    sys.stderr.write('Exit!\n')
    exit(1)

with open(output_config_path , 'w') as f:
  yaml.dump(config, f)

print('wrote config file to ' + output_config_path)
print('Finished.')

