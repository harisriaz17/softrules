# Soft Rules
Compute the matching score between a rule and a sentence using MPNet from [sentence transformers](https://www.sbert.net/docs/pretrained_models.html). In case of hard matching, the rule 
```
[word=was] [word=founded] [word=by]
```
matches the sentence
```
Microsoft was founded by Bill Gates
```
but does not match the sentence
```
Microsoft, founded by Bill Gates in his garage, is a very well-known company
```
We can notice though that the rule "almost" matches. The goal here is to give a numeric value to "almost".

## Architecture
<p align="center">
<img src="/docs/mpnet.png" alt="Architecture of our proposed method"/>
</p>

## Code

The structure is:
- `datagen.py` contains the base class for generating the `train`, `dev` and `test` partitions of the gold data. We can simply generate and save the data as follows (e.g. to generate positive-negative pairs from N=15000 and save them):
```python
   from datagen import DataGen
   dg = DataGen("train_tacred_old.tsv", "\t")
   dg.make_and_save_train_dev_test_splits(15000)
```
- The generated data is stored in the `/data` folder

- `train_mpnet.py` contains code for training MPNet on the background dataset. We can train like the following example:
    - ```python train_mpnet.py --model_config NO-custom-tokens --loss cosine```
    The trained model is saved in the `models/` folder. 
    
- `preprocess_5_way_1_shot.py` & `preprocess_5_way_5_shot.py` contains the code to preprocess the 1-shot and 5-shot settings from fewshot TACRED.
      You can run these scripts like this example: ```python preprocess_5_way_1_shot.py --model_name models/NO-custom-tokens-cosine```
      This will precompute & cache the embeddings in the fewshot_tacred/5_way_1_shot folder 
      
- Run `eval_5_way_1_shot.py` or `eval_5_way_5_shot.py` to evaluate the saved models on fewshot TACRED.
      You can run these scripts like this example: ```python eval_5_way_1_shot.py --model_name models/NO-custom-tokens-cosine```
      
- To evalate on the background dataset, you can do like this: ```python eval_background.py --model_name models/NO-custom-tokens-cosine```


### Installation
- `sentence-transformers` from [sbert.net](https://www.sbert.net/docs/installation.html)
- `odinson-gateway` from [lum-ai](https://github.com/lum-ai/odinson-gateway)
- `odinson-ruleutils` from [clu-ling](https://github.com/clu-ling/odinson-ruleutils)
- The rest of the dependencies, as specified in `environment.yml`. Note thet they are standard dependencies, which can be installed from standard channels

Note: If one of the libraries to be installed from github (`odinson-gateway`, `odinson-ruleutils`) fails due to missing the `setuptools` libraries, just install it with `conda install -c anaconda setuptools`.

### Results
Micro F1 scores on fewshot-TACRED:
| Fewshot TACRED setting  | No-custom-tokens-cosine | Some-custom-tokens-cosine  | No-custom-tokens-contrastive |
| ----------------------- | ----------------------- | -------------------------- | ---------------------------- |
| 5-way 1-shot            |        15.30            |            13.65           |             12.72            |          
| 5-way 5-shot            |        15.20            |            16.60           |             15.51            |
