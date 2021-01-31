import re
import string

corpus = open('../data/news1.txt').read()
corpus = [(re.sub(r'\d+', '', corpus.lower()).translate(str.maketrans('','', string.punctuation))).strip().split()]

