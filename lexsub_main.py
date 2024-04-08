#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

import gensim
import transformers 

from typing import List
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    
    lexemes = wn.lemmas(lemma, pos=pos)
    synsets = [lexeme.synset() for lexeme in lexemes]
    synonyms = []
    for synset in synsets:
        synonyms += [l.name() for l in synset.lemmas()]
        
    synonyms = set(synonyms)
    synonyms.discard(lemma)
    synonyms = [s.replace('_', ' ') for s in synonyms]
    
    return synonyms


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context : Context) -> str:
    
    lemma, pos = context.lemma, context.pos
    lexemes = wn.lemmas(lemma, pos=pos)
    synsets = [lexeme.synset() for lexeme in lexemes]
    counter = dict()
    for synset in synsets:
        for l in synset.lemmas():
            if l.name() != lemma:
                if l.name() not in counter:
                    counter[l.name()] = l.count()
                else:
                    counter[l.name()] += l.count()
    
    best = max(counter, key=counter.get)
    best = best.replace('_', ' ')
    return best


def wn_simple_lesk_predictor(context : Context) -> str:
    # could also implement: lemmatize context and definitions before comparison
    
    lemma, pos = context.lemma, context.pos
    full_context = set(context.left_context + context.right_context)
    lexemes = wn.lemmas(lemma, pos=pos)
    synsets = [lexeme.synset() for lexeme in lexemes]
    # synsets = [synset for synset in synsets if len(synset.lemmas()) > 1]
    # ^ could trim off the synsets containing only the target for efficiency,
    # but this would complicate finding the target's lemma in that synset later on
    # (would break the call to lexemes[i] in the final forloop as len(lexemes) != len(synsets) in that case)
    
    stop_words = stopwords.words('english')
    comparisons = synsets.copy()
    # Make each synset a list of itself and all its hypernyms, and for each synset of that list,
    # use the tokenized/normalized definition from the synset,
    # all tokenized/normalized examples for the synset,
    # make each token unique and remove any stopwords.
    for i in range(len(comparisons)):
        comparisons[i] = [synsets[i]] + synsets[i].hypernyms()
        comparisons[i] = [set(tokenize(synset.definition()) \
                            + tokenize(' '.join(synset.examples())) \
                              ).difference(stop_words)
                              for synset in comparisons[i]]
    
    overlaps = []
    for comparison in comparisons:
        overlap = 0
        for synset_comparison in comparison:
            overlap += len(set.intersection(full_context, synset_comparison))
        overlaps.append(overlap)
        
    # overlaps is now a list where overlap[i] corresponds to the overlap value of all lexemes in the synset synsets[i]
    assert len(overlaps) == len(synsets)
    
    candidates = [s.lemmas() for s in synsets]
    
    best = None
    best_val = 0
    
    for i, lemmas in enumerate(candidates):
        for l in lemmas:
            if l.name() != lemma:
                val = (1000 * overlaps[i]) \
                    + ( 100 * lexemes[i].count()) \
                    + (   1 * l.count())
                if val > best_val:
                    best = l.name().replace('_', ' ')
                    best_val = val
    
    return best
    

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        
        lemma, pos = context.lemma, context.pos
        
        # Copied code from part 1:
        lexemes = wn.lemmas(lemma, pos=pos)
        synsets = [lexeme.synset() for lexeme in lexemes]
        synonyms = []
        for synset in synsets:
            synonyms += [l.name() for l in synset.lemmas()]
            
        synonyms = set(synonyms)
        synonyms.discard(lemma)
        synonyms = [s for s in synonyms if '_' not in s] # ignore multi-word expressions
        
        synonyms = [s for s in synonyms if s in self.model.key_to_index] # ignore words not in the model
        
        nearest = max(synonyms, key=lambda s: self.model.similarity(lemma, s))
        return nearest


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        
        lemma, pos = context.lemma, context.pos
        
        # Copied code from part 1:
        lexemes = wn.lemmas(lemma, pos=pos)
        synsets = [lexeme.synset() for lexeme in lexemes]
        synonyms = []
        for synset in synsets:
            synonyms += [l.name() for l in synset.lemmas()]
            
        synonyms = set(synonyms)
        synonyms.discard(lemma)
        synonyms = [s for s in synonyms if '_' not in s] # ignore multi-word expressions
        
        vocab = self.tokenizer.get_vocab()
        synonyms = [s for s in synonyms if s in vocab] # ignore candidates that are not in the model's vocabulary
        
        sentence = ' '.join(context.left_context + ['[MASK]'] + context.right_context)
        input_toks = self.tokenizer.encode(sentence)
        target_i = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0][0, target_i] # an array where the index i is the score of the ith word of the vocab
        
        best = max(synonyms, key=lambda s: predictions[vocab[s]])
        return best


class MyPredictor(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    
    def valid(self, target, replacement):
        if target == replacement:
            return False
        if '_' in replacement:
            return False
        if replacement not in self.model.key_to_index:
            return False
        return True
    
    def get_example(self, lemma, synset):
        for e in synset.examples():
            if lemma in e:
                return e
        return lemma
    
    def score(self, target, context, replace, replace_ex, context_radius=1):
        """
        Score by both the cosine similarity to the replacement as well as the
        average cosine similarity between each word (context_radius) to the
        left and right of the word and the replacement in each example it has
        from WordNet
        """
        
        target_sim = self.model.similarity(target, replace)
        
        contexts = [[s.lower() for s in context.left_context][::-1], [s.lower() for s in context.right_context]]
        ex_tokens = tokenize(replace_ex)
        try:
            example = [ex_tokens[:ex_tokens.index(replace.lower())][::-1], ex_tokens[ex_tokens.index(replace.lower())+1:]]
        except ValueError:
            return target_sim
        
        context_sims = []
        for direction in range(2):
            for i in range(1, context_radius + 1):
                try:
                    c, e = contexts[direction][i], example[direction][i]
                    if c in self.model.key_to_index and e in self.model.key_to_index:
                        context_sims.append(self.model.similarity(c, e))
                except IndexError:
                    pass
        
        if len(context_sims) > 0:
            return target_sim + (sum(context_sims) + len(context_sims))
        else:
            return target_sim

    def predict(self, context : Context) -> str:
        
        lemma, pos = context.lemma, context.pos
        lexemes = wn.lemmas(lemma, pos=pos)
        synsets = [lexeme.synset() for lexeme in lexemes]
        candidates = []
        for s in synsets:
            for l in s.lemmas():
                if self.valid(lemma, l.name()):
                        candidates.append((l.name(), self.get_example(l.name(), s)))
        
        return max(candidates, key=lambda x: self.score(lemma, context, x[0], x[1]))[0]


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = MyPredictor(W2VMODEL_FILENAME)
    # predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
