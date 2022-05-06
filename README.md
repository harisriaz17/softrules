# Soft Rules
Compute the matching score between a rule and a sentence using MPNet from `sentence transformers`. In case of hard matching, the rule 
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

## Code

The structure is:
- `config` which contains some of the configs used to run different apps. There is one `config/base_config.yaml` which is user-specific. It contains the basepath. In this way the configs from the folder can work without additional modifications. Additionally, there is the folder `config/dataset_specifics` which contains information specific to each dataset (e.g. datasets may use different labels for the absence of a relation)
- `data_sample` contains a folder corresponding to each dataset supported. The idea is to provide a small file with the format of the dataset to facilitate runs.
- `src` contains the source code, organized as follows:
    - `src/apps` contains some simple runnables. The idea is to be able to take each individual file and run it without much orchestration. For example, a file might generate some baseline results
        - `src/apps/eval` contains the code for evaluation
    - `src/dataprocessing` contains the code to process the datasets supported. Each dataset is converted to an internal universal format
    - `src/model` contains the code for different models, such as baselines (e.g. word averaging) or proposed methods (e.g. transformers)
    - `src/rulegeneration` contains the code to generate rules from examples. One such approach can be to just use the words in between the entities. Other approaches include calling 3rd party models to generate rules


### Installation
- `odinson-gateway` from [lum-ai](https://github.com/lum-ai/odinson-gateway)
- `odinson-ruleutils` from [clu-ling](https://github.com/clu-ling/odinson-ruleutils)
- The rest of the dependencies, as specified in `environment.yml`. Note thet they are standard dependencies, which can be installed from standard channels

Note: If one of the libraries to be installed from github (`odinson-gateway`, `odinson-ruleutils`) fails due to missing the `setuptools` libraries, just install it with `conda install -c anaconda setuptools`.

