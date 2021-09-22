#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Very quick look at using word vectors with the built-in functions from
the gensim python library.
"""

import gensim.downloader as api

# load the word vectors file from the internet
glove = api.load("glove-wiki-gigaword-300")

# list the top n words that are most similar to the input
glove.most_similar('toaster', topn=16)

# the most_similar method can also solve analogies in the way we discussed
# in the other script: for example, to solve "toast is to toaster as pizza 
# is to blank", blank should be a word embedding that roughly solves
# toast - toaster ~= pizza - blank, i.e. blank ~= toaster - toast + pizza

glove.most_similar(
    positive=['toaster', 'pizza'],
    negative=['toast'],
    topn=5
)
