from pycocotools.coco import COCO
import nltk.tokenize
import numpy as np
from tqdm import tqdm
import json
import sys

def find_slurs(words, vocab):
    slurs = {}
    for word in words:
        if word in vocab:
            slurs[word] = vocab[word]
    return slurs

def make_vocab(anns):
    vocab = {}
    
    for ann in anns:
        tokenized = nltk.tokenize.word_tokenize(ann['caption'].lower())
        for word in tokenized:
            if word == '.':
                continue
            else:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1 
    words = list(vocab.keys())
    counts = list(vocab.values())
    return vocab, words, counts

def main(args):
    if len(args) < 2:
        raise Exception('Please include the vocab file you want to use') 

    slurs_txt = open('../annotations/{}'.format(args[1]), 'r') 
    words = [x.strip() for x in slurs_txt.readlines()]
    models = ['fc', 'd_d0', 'd_d10', 'transf', 'aoa', 'oscar_new']
    for m in models:
        for i in range(5):
            with open('../results/{}_{}.json'.format(m, i)) as f:
                res = json.load(f) 
            res_vocab, res_words, res_counts = make_vocab(res)
            res_slurs = find_slurs(words, res_vocab)
            print(m, res_slurs, i)

if __name__=="__main__":
    main(sys.argv)
