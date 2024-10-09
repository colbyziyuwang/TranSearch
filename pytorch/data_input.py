from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset

import config

from ast import literal_eval
import collections
from tqdm import tqdm
import itertools
import torch
from transformers import AutoModel, AutoTokenizer

# Supress warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("activebus/BERT_Review")
model = AutoModel.from_pretrained("activebus/BERT_Review")

# Function to get BERT embedding for a reviews
def get_bert_embedding(text):
    embeddings = [torch.zeros(768)]
    text = ' '.join(text)
    # Split text into chunks of 512 tokens
    for i in range(0, len(text), 512):
        chunk = text[i:i+512]
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the embedding from the [CLS] token (first token)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        embeddings.append(embedding)
    
    # Aggregate embeddings (e.g., by averaging)
    return torch.mean(torch.stack(embeddings), dim=0)

class PretrainData(Dataset):
	def __init__(self, neg_num):
		""" For pretraining the image and review features. """
		self.neg_num = neg_num
		self.asin_dict = json.load(open(config.asin_sample_path, 'r'))
		self.all_items = set(self.asin_dict.keys())

		# textual feture data
		# self.doc2vec_model = Doc2Vec.load(config.doc2model_path)
		# self.text_vec = {
			# asin: self.doc2vec_model.docvecs[asin] for asin in self.asin_dict}
		
		full_data = pd.read_csv(config.full_path)
		full_data.query_ = full_data.query_.apply(literal_eval)
		full_data.reviewText = full_data.reviewText.apply(literal_eval)

        # gather reviews to same asins
		raw_doc = collections.defaultdict(list)
		for k, v in zip(full_data.asin, full_data.reviewText):
			raw_doc[k].append(v)

		# concatenate the reviews together
		for k in raw_doc.keys():
			m = [i for i in raw_doc[k]]
			m = list(itertools.chain.from_iterable(m))
			raw_doc[k] = m

        # Replace the line where doc2vec is loaded with the BERT embedding process
		self.text_vec = {asin: get_bert_embedding(raw_doc[asin]) for asin in tqdm(self.asin_dict, desc="bert")}
		
		# visual feature data
		self.vis_vec = np.load(config.img_feature_path, allow_pickle=True).item()

	def sample_neg(self):
		""" Sample the anchor, positive, negative tuples. """
		self.features = []
		for asin in self.asin_dict:
			pos_items = self.asin_dict[asin]['positive']
			if not len(pos_items) == 0:
				for pos in pos_items:
					neg = np.random.choice(list(
							self.all_items - set(pos_items)),
							self.neg_num, replace=False)
					for n in neg:
						self.features.append((asin, pos, n))

	def __len__(self):
		""" For each anchor item, sample neg_number items."""
		return len(self.features)

	def test(self):
		for asin in self.asin_dict:
			anchor_text = self.text_vec[asin]
			anchor_vis = self.vis_vec[asin]
			yield (anchor_text, anchor_vis, asin)

	def __getitem__(self, idx):
		feature_idx = self.features[idx]
		anchor_item = feature_idx[0]
		pos_item = feature_idx[1]
		neg_item = feature_idx[2]

		anchor_vis = self.vis_vec[anchor_item]
		anchor_text = self.text_vec[anchor_item]
		pos_vis = self.vis_vec[pos_item]
		pos_text = self.text_vec[pos_item]
		neg_vis = self.vis_vec[neg_item]
		neg_text = self.text_vec[neg_item]

		sample = {'anchor_text': anchor_text, 
				'anchor_vis': anchor_vis,
				'pos_text': pos_text, 'pos_vis': pos_vis,
				'neg_text': neg_text, 'neg_vis': neg_vis}
		return sample


class TranSearchData(PretrainData):
	def __init__(self, neg_num, is_training):
		""" Without pre-training, input the raw data. """
		super().__init__(neg_num)
		self.is_training = is_training
		split = config.train_path if self.is_training else config.test_path
		self.data = pd.read_csv(split)

		self.query_dict = json.load(open(config.query_path, 'r'))
		self.user_bought = json.load(open(config.user_bought_path, 'r'))
		self.items = list(self.asin_dict.keys())

	def sample_neg(self):
		""" Take the also_view or buy_after_viewing as negative samples. """
		self.features = []
		for i in tqdm(range(len(self.data)), desc="data"):
			# query_vec = self.doc2vec_model.docvecs[
								# self.query_dict[self.data['query_'][i]]]
			query_vec = get_bert_embedding(self.data['query_'][i])

			if self.is_training:
				# We tend to sample negative ones from the also_view and 
				# buy_after_viewing items, if don't have enough, we then 
				# randomly sample negative ones.
				asin = self.data['asin'][i]
				sample = self.asin_dict[asin]
				all_sample = sample['positive'] + sample['negative']
				negs = np.random.choice(
					all_sample, self.neg_num, replace=False, p=sample['prob'])
			
				self.features.append(((
						self.data['userID'][i], query_vec), (asin, negs)))

			else:
				self.features.append(((self.data['userID'][i], query_vec),
					(self.data['reviewerID'][i], self.data['asin'][i]),
					self.data['query_'][i]))

	def get_all_test(self):
		for asin in self.asin_dict:
			sample_vis = self.vis_vec[asin]
			sample_text = self.text_vec[asin]
			yield sample_vis, sample_text, asin

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		feature_idx = self.features[idx]
		userID = feature_idx[0][0]
		query = feature_idx[0][1]

		if self.is_training:
			pos_item = feature_idx[1][0]
			neg_items = feature_idx[1][1]

			pos_vis = self.vis_vec[pos_item]
			pos_text = self.text_vec[pos_item]

			neg_vis = [self.vis_vec[i] for i in neg_items]
			neg_text = [self.text_vec[i] for i in neg_items]
			neg_vis = np.array(neg_vis)
			neg_text = np.array(neg_text)

			sample = {'userID': userID, 'query': query,
					  'pos_vis': pos_vis, 'pos_text': pos_text,
					  'neg_vis': neg_vis, 'neg_text': neg_text}

		else:
			reviewerID = feature_idx[1][0]
			item = feature_idx[1][1]
			query_text = feature_idx[2]

			sample = {'userID': userID, 'query': query,
					  'reviewerID': reviewerID, 'item': item,
					  'query_text': query_text}
		return sample
