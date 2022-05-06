import argparse
import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import evaluation
from datagen import DataGen
import pickle

def main(settings):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = settings.num_gpu
	
	# create and save train, dev, test data
	num_samples = 15000	
	dg = DataGen("train_tacred_old.tsv", '\t')
	dg.make_and_save_train_dev_test_splits(settings.data_path, num_samples)

	# open train list 
	open_file = open(settings.data_path+"train", "rb")
	train = pickle.load(open_file)

	# open dev list 
	open_file = open(settings.data_path+"dev", "rb")
	dev = pickle.load(open_file)
	
	# construct train dataloader
	train_examples = []
	for i in range(len(train)):
		input_example = InputExample(texts=[train[i][0], train[i][1]], label=train[i][3])
		train_examples.append(input_example)

	
	# construct dev set
	dev_sentences = []
	dev_rules = []
	dev_relations = []
	# equivalent to labels for dev (cosine similarity scores)
	dev_scores = [] 

	for i in range(len(dev)):
		dev_sentences.append(dev[i][0])
		dev_rules.append(dev[i][1])
		dev_relations.append(dev[i][2])
		dev_scores.append(dev[i][3])
	
	NUM_EPOCH = settings.epochs
	model = SentenceTransformer('all-mpnet-base-v2')
	train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=16)
	if settings.loss == "cosine":
		train_loss = losses.CosineSimilarityLoss(model)
	else:
		train_loss = losses.ContrastiveLoss(model)		
	
	'''
	Evaluate a model based on the similarity of the embeddings by calculating the Spearman correlation with cosine similarity in comparison to the gold labels of the dev set
	writes the results on dev to a csv file
	'''
	evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_sentences, dev_rules, dev_scores, main_similarity = evaluation.SimilarityFunction.COSINE, 
	show_progress_bar=True, write_csv=True)
	
	if settings.model_config == "Some_custom_tokens":
		# add extra token(s) to transformer vocabulary
		extra_tokens = ['misc', 'criminal_charge', 'cause_of_death', 'url', 'state_or_province']
		model._first_module().tokenizer.add_tokens(extra_tokens, special_tokens=False)
		model._first_module().auto_model.resize_token_embeddings(len(model._first_module().tokenizer))
	
	elif settings.model_config == "ALL-custom-tokens":
		# add extra token(s) to transformer vocabulary
		extra_tokens_subj = ['subj-person', 'subj-location', 'subj-organization', 'subj-misc', 'subj-money', 'subj-number', 'subj-ordinal', 'subj-percent', 'subj-date', 
		'subj-time', 'subj-duration', 'subj-set', 'subj-email', 'subj-url', 'subj-city', 'subj-state_or_province', 'subj-country', 'subj-nationality', 'subj-religion', 
		'subj-title', 'subj-ideology', 'subj-criminal_charge', 'subj-cause_of_death', 'subj-handle']

		extra_tokens_obj = ['obj-person', 'obj-location', 'obj-organization', 'obj-misc', 'obj-money', 'obj-number', 'obj-ordinal', 'obj-percent', 'obj-date', 'obj-time', 
		'obj-duration', 'obj-set', 'obj-email', 'obj-url', 'obj-city', 'obj-state_or_province', 'obj-country', 'obj-nationality', 'obj-religion', 'obj-title', 
		'obj-ideology', 'obj-criminal_charge', 'obj-cause_of_death', 'obj-handle']
		
		extra_tokens = extra_tokens_subj + extra_tokens_obj
		model._first_module().tokenizer.add_tokens(extra_tokens, special_tokens=False)
		model._first_module().auto_model.resize_token_embeddings(len(model._first_module().tokenizer))		
		
	model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=NUM_EPOCH, evaluator=evaluator, evaluation_steps = 1000, warmup_steps=100, 
	output_path=settings.model_save_path+settings.model_config+settings.loss+"15k", show_progress_bar=True)		

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=str, default="0")
    parser.add_argument('--model_config', type=str, default="NO-custom-tokens")
    parser.add_argument('--model_save_path', type=str, default="models/")
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--loss', type=str, default="cosine")
    

    settings = parser.parse_args()
    main(settings)
