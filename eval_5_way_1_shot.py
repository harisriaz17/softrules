import argparse
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle
from tqdm import tqdm
from metrics import *


def main(settings):
	if settings.model_name == "pretrained":
		# just use the pretrained MPNet
		model = SentenceTransformer('all-mpnet-base-v2')
	else:
		model = SentenceTransformer(settings.model_name)		
		
	# load required episodes_rule_sim dict from disk
	with open(settings.cache_path+"episodes_rule_sim.pickle", "rb") as inFile:
		episodes_rule_sim = pickle.load(inFile)
 
	# load required rule_to_relation dict from disk
	with open(settings.cache_path+"fewshot_rule_to_relation.pickle", "rb") as inFile:
   		rule_to_relation = pickle.load(inFile)
   	
	# load required gold relation list from disk (first 10,000)
	with open(settings.cache_path+"fewshot_gold_relations.pickle", "rb") as inFile:
		gold_relations = pickle.load(inFile)
	
	print("\n--------------------Testing Model--------------------\n")
	
	thresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
	f1_for_thresh = []
	for t in thresh:
		# for every threshold we have different set of pred_relations
		pred_relations = []
		for episode in range(0, 30000):
			rule_sim_pairs = episodes_rule_sim[episode] # list of (rule, sim) tuples
			max_sim = 0.0
			for pair in rule_sim_pairs:
				if(pair[1] > t):
					max_sim = pair[1]
					best_rule = pair[0]
		
			# no rule in rule_sim_pairs matched with a cosine similarity above threshold (max_sim is still 0)
			if max_sim == 0.0:
				pred_relations.append(NO_RELATION)
				continue
			else:
				pred_relations.append(rule_to_relation[best_rule])
	
		print("\n\nFor threshold = ", t, "\n")
		prec_micro, recall_micro, f1_micro = tacred_score(gold_relations, pred_relations, True)
		f1_for_thresh.append((t, prec_micro, recall_micro, f1_micro))
	
	print("Performance on ALL 30,000 episodes: ", f1_for_thresh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=str, default="0")
    parser.add_argument('--model_name', type=str, default="pretrained")
    parser.add_argument('--cache_path', type=str, default="fewshot_tacred/5_way_1_shot/")
    settings = parser.parse_args()
    main(settings)
