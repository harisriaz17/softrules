import argparse
import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle
from tqdm import tqdm
from metrics import *
from datagen import DataGen

def main(settings):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = settings.num_gpu
	
	dg = DataGen("train_tacred_old.tsv", '\t')
	rules_to_relations = dg.rules_to_relations

    # open test list
	open_file = open(settings.data_path+"test", "rb")
	test = pickle.load(open_file)
	
	# construct test set
	test_sentences = []
	test_rules = []
	test_relations = []
	# equivalent to test labels (cosine similarity scores)
	test_scores = [] 
	
	for i in range(len(test)):
		test_sentences.append(test[i][0])
		test_rules.append(test[i][1])
		test_relations.append(test[i][2])
		test_scores.append(test[i][3])
	
	if settings.model_name == "pretrained":
		# just use the pretrained MPNet
		model = SentenceTransformer('all-mpnet-base-v2')
	else:
		model = SentenceTransformer(settings.model_name)		
	
	print("\n--------------------Testing Model--------------------\n")

	rule_to_embedding = {}	
	tsent_to_embedding = {}
	assigned_relations = []	
	thresh = settings.thresh
	for i in tqdm(range(len(test_sentences))):	
		if test_sentences[i] in tsent_to_embedding:
			testsenti_embedding = tsent_to_embedding[test_sentences[i]]
		else:
			testsenti_embedding = model.encode(test_sentences[i])
			tsent_to_embedding[test_sentences[i]] = testsenti_embedding
	
		for j in range(len(test_rules)):
			if test_rules[j] in rule_to_embedding:
				rulesj_embedding = rule_to_embedding[test_rules[j]]
			else:
				rulesj_embedding =  model.encode(test_rules[j])
				rule_to_embedding[test_rules[j]] = rulesj_embedding	
			
			sim = cos_sim(rulesj_embedding, testsenti_embedding)
			if sim > thresh:
				assigned_rule = test_rules[j]
				assigned_rel = rules_to_relations[assigned_rule]
				break
			else: 
				assigned_rel = NO_RELATION
	
		assigned_relations.append(assigned_rel)
	
	print("Accuracy on test set: ", accuracy_percent(test_relations, assigned_relations))
	print(tacred_score(test_relations, assigned_relations))
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=str, default="0")
    parser.add_argument('--model_name', type=str, default="pretrained")
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--thresh', type=float, default=0.99)
    settings = parser.parse_args()
    main(settings)
    

