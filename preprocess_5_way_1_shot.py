import argparse
import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle
from tqdm import tqdm
from metrics import *

def main(settings):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = settings.num_gpu
	episodes = []
	with open(settings.data_path, "r") as ins:
		for line in ins:
			episode = json.loads(line)
			episodes.append(episode)

	if settings.model_name == "pretrained":
		# just use the pretrained MPNet
		model = SentenceTransformer('all-mpnet-base-v2')
	else:
		model = SentenceTransformer(settings.model_name)		
	
	print("\n--------------------Preprocessing Fewshot TACRED 5-way 1-shot--------------------\n")
	
	gold_relations = []
	
	episodes_rule_sim = defaultdict(list)
	rule_to_relation = defaultdict(str)
	rule_to_embedding = {}
	tsent_to_embedding = {}


	for episode in tqdm(range(0, 30000)):
		# print("Processing episode: ", episode)
		gold_relations.append(episodes[episode]["gold_relation"])
	
		''' in each episode, this list holds the best candidate rules i.e. supporting rules in an episode 
			that match the test sentence of that episode with cosine similarity score over a certain 
			threshold as a tuple: (rule, cosine_similarity_score)
		'''
		episode_rules = list(episodes[episode]["rules"])
		# construct supporting rule to relation dict
		for i in range(len(episode_rules)):
			rule_to_relation[episode_rules[i]] = episodes[episode]["rules_relations"][i]
	
		# test sentence in episode
		test_sentence = ' '.join(str(item) for item in episodes[episode]["test_sentence"])

		for i in range(len(episode_rules)):
			if episode_rules[i] in rule_to_embedding:
				episode_rulesi_embedding = rule_to_embedding[episode_rules[i]]
			else:
				episode_rulesi_embedding = model.encode(episode_rules[i])
				rule_to_embedding[episode_rules[i]] = episode_rulesi_embedding
		
			if test_sentence in tsent_to_embedding:
				testsenti_embedding = tsent_to_embedding[test_sentence]
			else:
				testsenti_embedding = model.encode(test_sentence)
				tsent_to_embedding[test_sentence] = testsenti_embedding
		
			
			sim = cos_sim(episode_rulesi_embedding, testsenti_embedding)
			'''
				save: {episode_num : [(support_rule1, sim1), (support_rule2, sim2) ... (support_rule_30k, sim_30k)]
				where sim_i is the similarity of rule_i with test_sentence corresponding to THAT episode_num
			'''
			episodes_rule_sim[episode].append((episode_rules[i], sim))

	# cache embeddings for fast evaluation on 5-way 1-shot 
	with open('fewshot_tacred/5_way_1_shot/episodes_rule_sim.pickle', 'wb') as fileOut:
		pickle.dump(episodes_rule_sim, fileOut)

	with open('fewshot_tacred/5_way_1_shot/rule_to_relation.pickle', 'wb') as fileOut:
		pickle.dump(rule_to_relation, fileOut)
	
	with open('fewshot_tacred/5_way_1_shot/gold_relations.pickle', 'wb') as fileOut:
		pickle.dump(gold_relations, fileOut)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=str, default="0")
    parser.add_argument('--model_name', type=str, default="pretrained")
    parser.add_argument('--data_path', type=str, default="fewshot_tacred/5_way_1_shot/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl")
    settings = parser.parse_args()
    main(settings)
    

