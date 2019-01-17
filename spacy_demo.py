# Code for obtaining sentence vector from spacy module

import spacy
nlp = spacy.load('en')
tweet_doc = nlp("Here goes your some input text")
print(tweet_doc.vector)
