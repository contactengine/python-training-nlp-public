# Intro to NLP in Python

A git repo for the files we're working on during the intro to NLP sessions we're running as part of the Python training.

## Problem

We're trying to build a simple classifier to predict intents in a text dataset based on crowdsourced virtual assistant requests.

## Session 1

We looked at word vectors, a fundamental concept in modern NLP that provides a way to read words (more specifically called tokens in this context, i.e. parts of words/combinations of characters arising in our text) and transform them into vectors of numbers for a machine learning model to ingest, in a way that maintains certain relationships (e.g. semantic or grammatical) between them.

For the [Gensim](https://radimrehurek.com/gensim/) part, you'll need to install it by running (in the Anaconda command line, preferably in a new conda environment that you've set up just for this project):

    pip install gensim

The other modules used so far should all be present in any base installation of Anaconda.

## Session 2

We made a set of features to act as input to a machine learning model, by using word vectors replacing the words in the data with word vectors and then aggregating them in a couple of different ways.

## Files

* **word_vectors.py**: Session 1 word vector exploration
* **gensim_word_vectors.py**: word vectors using the gensim package
* **process.py**: data processing script for the intent classification data
* **glove/**: glove embedding text files, needs to be downloaded separately from <https://nlp.stanford.edu/projects/glove/>
* **oos_data/**: CLINIC public intent classification dataset, downloaded separately from <https://github.com/clinc/oos-eval/>
* **classifier.py**: Session 2 and 3, using word vectors to turn a text dataset into numerical features for an intent classifier, then (to come in Session 3) training and testing that classifier

## Additional material

* [Tensorflow Embedding Projector](https://projector.tensorflow.org/): projections into three- and two-dimensional space of word embeddings, giving you a nice way of visualising clusters/similarity/etc. of word vector sets
* [Man is to Computer Programmer as Woman is to Homemaker?
Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520.pdf): the paper we discussed looking at gender bias in word embeddings, as well as ideas for how to get rid of some of it
