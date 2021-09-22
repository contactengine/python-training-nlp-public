#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Loading and using the GloVe word vectors from first principles.
"""

import numpy as np

# setting a variable at the top of the script so that we can change the
# dimension at any time
glove_dim = 300

# load the glove vectors line by line as a python dictionary
glove = {}
with open(f'./glove/glove.6B.{glove_dim}d.txt', 'r') as f:
    for line in f:
        split = line.split()
        glove[split[0]] = np.array(split[1:], dtype='float64')

# our complete word dictionary for the GloVe word embeddings is just the
# list of keys of our dictionary
glove_index = list(glove.keys())


def cos_sim(v, w):
    """Cosine similarity function measuring how "close" or
    "similar" two word vectors are.
    """
    return np.dot(v, w)/(np.linalg.norm(v) * np.linalg.norm(w))


# for example, "computer" should be close to "pc"
cos_sim(glove['computer'], glove['pc'])
# but not very close to "dog"
cos_sim(glove['computer'], glove['dog'])


# we can also use word vectors to solve simple "analogies" of the form
# A is to B as C is to BLANK; the following is a function that does just
# that in a fairly naive way
def analogy(A, B, C, word_vectors=glove):
    """Iterate through word_vectors in order to find one whose relationship
    to C is similar to that of B to A. In practice, we're looking to solve
    the word vector "equation" A - B ~= C - BLANK, i.e. BLANK ~= -A + B + C.
    """
    max_similarity = -10
    most_similar = 'N/A'

    for word, vector in word_vectors.items():
        if word in [A, B, C]:
            continue
        similarity = cos_sim(
            - word_vectors[A] + word_vectors[B] + word_vectors[C],
            vector
        )
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = word
    return most_similar, max_similarity


# for example, this solves "paris is to france as london is to BLANK"
analogy('paris', 'france', 'london')
# this is the "canonical" example everyone gives
analogy('man', 'woman', 'king')
# example of gender bias present in the GloVe embeddings (see paper in
# README.md)
analogy('doctor', 'nurse', 'man')
