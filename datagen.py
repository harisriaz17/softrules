import pandas as pd
from collections import Counter, defaultdict
import pickle
from typing import Dict, List
import random
import math
import os

class DataGen:
	def __init__(self, path, sep):
		self.path = path
		self.sep = sep
		self.raw_data = self.readData(self.path, self.sep)
		self.rules = self.raw_data['pattern'].values
		self.relations = self.raw_data['relation'].values
		self.sentences = list(self.raw_data['sentence'])
		self.relation_to_rules = self.relationToRules(self.rules, self.relations, self.sentences)
		self.relation_to_sents = self.relationToSents(self.rules, self.relations, self.sentences)
		self.unique_rels = self.getUniqueRelations(self.relations)
		self.rules_to_relations = self.rulesToRelations(self.rules, self.relations)
	
	
	def readData(self, path, sep='\t'):
		return pd.read_csv(path, sep=sep)
	
	def getRules(self, data):
		return data['pattern'].values 
	
	def getRelations(self, data):
		return data['relation'].values
	
	def getSentences(self, data):
		return list(data['sentence'])
		
	def relationToRules(self, rules, relations, sentences):
		relation_to_rules = defaultdict(list)
		for (rule, relation, sentence) in zip(rules, relations, sentences):
			relation_to_rules[relation].append(rule)
		
		return relation_to_rules
			
	def relationToSents(self, rules, relations, sentences):
		relation_to_sents = defaultdict(list)
		for (rule, relation, sentence) in zip(rules, relations, sentences):
			relation_to_sents[relation].append(sentence)
		
		return relation_to_sents
	
	'''
		construct a rule to relation dict to be used at eval time for fetching the gold relation for every rule
	'''
	def rulesToRelations(self, rules, relations):
		rules_to_relations = defaultdict(str)
		for (rule, relation) in zip(rules, relations):
			rules_to_relations[rule] = relation
			
		return rules_to_relations
		
	def getUniqueRelations(self, relations):
		return list(set(relations))
	
	'''
		Method to generate gold data from background dataset using our special mix and match scheme
	'''
	def generate_gold_data(self, num_examples=None):
		if not num_examples:
			print("Attribute Error: num_examples is empty")
			return
		else:
			gold_data = []
			
			for _ in range(num_examples):
				# constructing a positive example
				random_rel1 = random.choice(self.unique_rels) # this is rel 1
				random_rule1 = random.choice(list(self.relation_to_rules[random_rel1]))
				random_sent1 = random.choice(list(self.relation_to_sents[random_rel1]))
				gold_data.append((random_sent1, random_rule1, random_rel1, 1.0))
		
				# constructing a negative example
				random_rel2 = random.choice(self.unique_rels) # this is rel 2
				while(random_rel2 == random_rel1):
					random_rel2 = random.choice(self.unique_rels) # this is rel 2
				random_rule2 = random.choice(list(self.relation_to_rules[random_rel2]))
				random_sent2 = random.choice(list(self.relation_to_sents[random_rel2]))		
	
				gold_data.append((random_sent2, random_rule1, random_rel2, 0.0))
		
		
			return gold_data
	
	
	def make_and_save_train_dev_test_splits(self, data_save_path="data/", num_examples=None):
		if not num_examples:
			print("Attribute Error: num_examples is empty")
			return
		else:
			gold_data = self.generate_gold_data(num_examples)
			# choose 70% of gold data for training
			TRAIN_SIZE = int(math.floor(0.7 * len(gold_data)))
			train = gold_data[:TRAIN_SIZE]
			# choose 10% (one-thirds) of the remaining 30% as dev
			DEV_SIZE = int((1/3)* (len(gold_data) - (0.7*len(gold_data))))
			dev = gold_data[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE]
			# choose 20% (two-thirds) of the remaining 20% for test
			test = gold_data[TRAIN_SIZE+DEV_SIZE:]
			
			save_path = data_save_path
			
			# make directory for saving data if not exists
			if not os.path.exists(data_save_path):
				os.makedirs(data_save_path)
				
			# save train list
			open_file = open(save_path+"train", "wb")
			pickle.dump(train, open_file)
			
			# save dev list
			open_file = open(save_path+"dev", "wb")
			pickle.dump(dev, open_file)
			
			# save dev list
			open_file = open(save_path+"test", "wb")
			pickle.dump(test, open_file)
			return
