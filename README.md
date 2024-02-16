
# ReTRF


## Guide

- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Config File](#config_file)

## Setup

```
conda env create -f environment.yml
```

## Training

### Training Vanilla NER Models (A1), same as CLNER (w/o context)
```bash
python train.py --config config/wnut171.yaml
```

### Training Models to reproduce the work from CLNER

Run:

```bash
python train.py --config config/wnut17_doc.yaml
```

### Training Models to reproduce the work of CLNER (+CL)

Run:

```bash
python train.py --config config/wnut17_doc_cl_kl.yaml
python train.py --config config/wnut17_doc_cl_l2.yaml
```

### Training Models for CLNER (w/ Our Sents)

```bash
python train.py --config config/wnut17_nonlocal_v1.yaml
```

### Training ReTRF Models

```bash
python train.py --config "config/A21.yaml"
python train.py --config "config/A21->A22.yaml"
python train.py --config "config/A21->A22->A23.yaml"
```

The following yaml file is used to configure the experiment.

```yaml
targets: ner
ner:
  Corpus: ColumnCorpus-1
  ColumnCorpus-1: 
    data_folder: datasets/conll_03_english
    column_format:
      0: text
      1: pos
      2: chunk
      3: ner
    tag_to_bioes: ner
  tag_dictionary: resources/taggers/your_ner_tags.pkl
```


The `tag_dictionary` is a path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically. The dataset format is: `Corpus: $CorpusClassName-$id`, where `$id` is the name of datasets (anything you like). You can train multiple datasets jointly. For example:

Please refer to [Config File](#Config-File) for more details.

### Train On a Recovery Dataset


The `$config_file` should have the following format
```yaml
targets: ner
ner:
  Corpus: ColumnCorpus-WNUTDOCRECOVERY
  ColumnCorpus-WNUTDOCRECOVERY:
    comment_symbol: '##'
    column_format:
      0: text
      1: ner  # gold label. B-X special token in this column will be masked in loss calculation.
      2: predict  # first stage prediction
      3: score_encoding  # score_0, score_1, ...
    data_folder: ...
    tag_to_bioes: ner
    scope_token_map:
      local_eos: <EOS>
      nonlocal_bos: <s>
      nonlocal_eos: <EOS>
  tag_dictionary: resources/taggers/aug_wnut_ner_tags.pkl
```

```
### nkj_silver
@paulwalk	O	O	score_3
It	O	O	score_5
's	O	O	score_5
the	O	O	score_6
...
Empire	B-location	B-location	score_0
State	I-location	I-location	score_0
Building	I-location	E-location	score_4
=	O	O	score_2
ESB	B-location	O	score_0
last	O	O	score_5
evening	O	O	score_5
.	O	O	score_4
<EOS>	B-X	<EOS>	<EOS>  ===> mapped to `local_eos`
<s> B-X <s> <s>  ===> mapped to `nonlocal_bos`
#### nss_silver  <== sentence-level labels
##### nts_silver  <== token-level labels
Empire	B-X	B-location	score_1
##### nts_silver
State	B-X	I-location	score_8
##### nts_silver
Building	B-X	E-location	score_9
##### 
-	B-X	O	score_2
##### 
ASCE	B-X	B-location	score_2
##### 
...
<EOS>	B-X	<EOS>	<EOS>  ===> mapped to `nonlocal_eos`
#### 
##### 
Top	B-X	O	score_3
##### 
50	B-X	O	score_4
##### 
Photos	B-X	O	score_7
...
<EOS>	B-X	<EOS>	<EOS>
#### nss_silver
##### nts_silver
Empire	B-X	B-location	score_3
##### nts_silver
State	B-X	I-location	score_9
##### nts_silver
Building	B-X	E-location	score_9
##### 
in	B-X	O	score_4
##### 
New	B-X	B-location	score_9
...
## id: 1  <--- lines have `$comment_symbol` as the 1st column will be ignored in dataset processing.
## nonlocals: 16
## tokens: 340
## subtokens: 509
## is_augmental: False
## augmented_by: None
## is_preserved: False
## example_type: None

```

Loading a dataset with the above format gives you the following output, where each token has its token-level labels, and each sentence has its sentence-level labels too. Also note that each token will belong to one scope from 
  1. "local_token": the tokens of the local sentence
  2. "local_eos": the end of the local sentence, which also denotes the boundary of the local and non-local sentences.
  3. "nonlocal_token": the tokens of non-local sentences
  4. "nonlocal_bos": the beginning of one non-local sentence
  5. "nonlocal_eos": the end of one non-local sentence.

```
sent labels:
[]
Token: 1 @paulwalk  (text) O (ner) O (predict) score_3 (score_encoding)          | labels: []   scope: local_token
Token: 2 It  (text) O (ner) O (predict) score_5 (score_encoding)         | labels: []   scope: local_token
Token: 3 's  (text) O (ner) O (predict) score_5 (score_encoding)         | labels: []   scope: local_token
Token: 4 the  (text) O (ner) O (predict) score_6 (score_encoding)        | labels: []   scope: local_token
Token: 5 view  (text) O (ner) O (predict) score_4 (score_encoding)       | labels: []   scope: local_token
Token: 6 from  (text) O (ner) O (predict) score_6 (score_encoding)       | labels: []   scope: local_token
...
Token: 28 <EOS>  (text) S-X (ner) <EOS> (predict) <EOS> (score_encoding)         | labels: []   scope: local_eos
Token: 29 Empire  (text) S-X (ner) B-location (predict) score_1 (score_encoding)         | labels: []   scope: nonlocal_token
Token: 30 State  (text) S-X (ner) I-location (predict) score_8 (score_encoding)          | labels: []   scope: nonlocal_token
...
Token: 399 -  (text) S-X (ner) O (predict) score_3 (score_encoding)      | labels: []   scope: nonlocal_token
Token: 400 The  (text) S-X (ner) O (predict) score_1 (score_encoding)    | labels: []   scope: nonlocal_token
Token: 401 Heart  (text) S-X (ner) O (predict) score_0 (score_encoding)          | labels: []   scope: nonlocal_token
Token: 402 and  (text) S-X (ner) O (predict) score_4 (score_encoding)    | labels: []   scope: nonlocal_token
Token: 403 Soul  (text) S-X (ner) O (predict) score_1 (score_encoding)   | labels: []   scope: nonlocal_token
Token: 404 of  (text) S-X (ner) O (predict) score_2 (score_encoding)     | labels: []   scope: nonlocal_token
```

## Evaluation

### Evaluation Mode 1 - evaluate by calling the model to generating predictions for the input file (CLI)

```bash
python train.py --test
python train.py --test --test_on_subsets train,dev,test  # test on the specific subsets like dev,test (separated by comma)
python train.py --test --test_on_subsets dev,test --all_tag_prob  # print the probabilities of all tags in the output file
```

### Evaluation Mode 2 - evaulate the given output prediction file (CLI)
```bash
python train.py --only_eval
```

### Prediction Mode - generate the model output prediction file (CLI)
```bash
python train.py --predict
```


### Inference Mode (CLI)
```bash
python train.py --inference --interactive --interactive_verbose
```
 * When inference mode is activated, you can load one model and type your own input in a default file located at `{target_dir}/{model_name}/inference/input.txt`.
 * Note that for loading a model, use `output_dir` and `model_name` in the config file, while the parameter `load_pretrained` and `pretrained_model` will be ignored.
 * Use `--interactive` to activate interactive mode
 * Use `--interactive_verbose` along with `--interactive` to give more detailed results like token-by-token probabilities.


## Config File

The config files are based on yaml format.

* `targets`: The target task
  * `ner`: named entity recognition
  * `upos`: part-of-speech tagging
  * `chunk`: chunking
  * `ast`: abstract extraction
  * `dependency`: dependency parsing
  * `enhancedud`: semantic dependency parsing/enhanced universal dependency parsing
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
  * `Corpus`: The training corpora for the model, use `:` to split different corpora.
  * `tag_dictionary`: A path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
  * `FastSequenceTagger`: Sequence labeling model. The values are the parameters.
  * `SemanticDependencyParser`: Syntactic/semantic dependency parsing model. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters, see `flair/embeddings.py` for more details. For each embedding, use `$classname-$id` to represent the class. For example, if you want to use BERT and M-BERT for a single model, you can name: `TransformerWordEmbeddings-0`, `TransformerWordEmbeddings-1`.
  * `TransformerWordEmbeddings-*`:
    * `custom_embeddings_params`: adding this attribute will create custom embedding layer
      * `<input the name of your custom embedding>`:
        * `vocab`: define the list of vocabulary here, or use `vocab_path`.
        * `vocab_path`: load vocabulary from a pickle file, or use `vocab`.
        * `from_pretrained`: load pretrained weights for the custom embeddings from a pickle file
        * `additional_special_tokens`: the list of tokens to be added missing in the pretrained custom embeddings (`from_pretrained`). These tokens will be trained from scratch. E.g., `['<EOS>', '<MASK>']`.
        * `use_different_eos`: `true` if using a different eos token is desired; othewise, `false`.
        * `params`: parameter dictionary to input to the embedding layer as `**kwargs`, for example,  `embedding_dim: 300`
        * `scale_pretrained`: the scale factor (float) of the pretrained embedding
        * `init_factor`: the scale factor of the initialized parameters of the embeddings. Default to 1.0.
        * `init_affine_weight`: 1.0
      * `merge_custom_embeddings`: how the custom embeddings are merged together with Transformer embeddings. Options: `add`, `concat`
      * `init_custom_embeddings`: how to initialize the custom embeddings: `zero`, `random`, `uniform`
      * `init_custom_embeddings_std`: the standard deviation of the embedding initialization
* `trainer`: The trainer class.
  * `ModelFinetuner`: The trainer for fine-tuning embeddings or simply train a task model without ACE.
    * `main_metric`: only_ex_w_nlc, only_ex_w_lc, ner  
  * `ReinforcementTrainer`: The trainer for training ACE.
* `train`: the parameters for the `train` function in `trainer` (for example, `ReinforcementTrainer.train()`).
* `load_pretrained`: load a pretrained model from the path specified in `pretrained_model`
* `pretrained_model`: see above.
