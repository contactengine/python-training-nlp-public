#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re

# setting a variable at the top of the script so that we can change the
# dimension at any time
glove_dim = 50

# load the glove vectors line by line as a python dictionary
glove = {}
with open(f'./glove/glove.6B.{glove_dim}d.txt', 'r') as f:
    for line in f:
        split = line.split()
        glove[split[0]] = np.array(split[1:], dtype='float64')

# our complete word dictionary for the GloVe word embeddings is just the
# list of keys of our dictionary
glove_index = list(glove.keys())
glove_matrix = np.vstack(glove.values())

df = pd.read_csv('./data.csv')

df.intent.value_counts()
df.category.value_counts()

df.sample()

df.text = df.text.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
text_data = df.text.str.split(' ').apply(lambda x: [s for s in x if x != ''])

glove_ood = glove_matrix.mean(axis=0)
text_embeddings = text_data.apply(lambda x: [glove.get(w, glove_ood) for w in x])

stacked_embeddings = text_embeddings.apply(np.vstack)
max_embeddings = stacked_embeddings.apply(lambda x: np.max(x, axis=0))
min_embeddings = stacked_embeddings.apply(lambda x: np.min(x, axis=0))
sentence_embeddings = pd.DataFrame({'min' : min_embeddings, 'max': max_embeddings}).apply(
    lambda row: np.hstack([row['min'], row['max']]), axis=1
)

