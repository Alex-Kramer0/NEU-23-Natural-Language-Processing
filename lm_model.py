from collections import Counter
import numpy as np
import math

"""
CS 4/6120, Fall 2023
Homework 3
Alex Kramer
"""

# Constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT 
   
    # Slices the list into n length lists and appends them to create our n-gram list
    
    output = []

    if n > len(tokens):
        # if the n grams is the same length as the list return the original list
        return tokens

    else:
        for i in range(len(tokens) - n + 1):
            # for each token in the list

            grouped = tokens[i:i + n]
            # create a new list called group, that slices the needed tokens
            # from the original list
            output.append(grouped)

        return output

def read_file(path: str) -> list:
    """
    Reads the contents of a file in line by line.
    Args:
    path (str): the location of the file to read

    Returns:
    list: list of strings, the contents of the file
    """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
    """
    Tokenize a single string. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
    False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

    Returns:
    list of strings - a single line tokenized
    """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
    # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end] #'<s>, <\s>
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
    """
    Tokenize each line in a list of strings. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

    Returns:
    list of strings - all lines tokenized as one large list
    """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        # STUDENTS IMPLEMENT
        
        # import
        from collections import Counter, defaultdict
        
        # initialize variables to store counts, totals, and ngrams in
        self.N = n_gram 
        
        # dict: {Prefix: value}
        self.prefix_counts = defaultdict(int)
        
        # dict of dicts --> {Word2: (Word1, ), Count}
        self.n_grams_counts = defaultdict(lambda: defaultdict(int)) 
        self.generate_n_grams = defaultdict(lambda: defaultdict(int)) 
        
        # known words
        self.knowns = set()
        # unknown words
        self.unknowns = [] 
        # sentence starts
        self.starts = [] 
        # vocab size
        self.V = 0 
        # adjusted in train
        self.counts = None
        
        
    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          tokens (list): tokenized data to be trained on as a single list
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """ 
        # STUDENTS IMPLEMENT
        
        # count occurance of each token
        self.counts = Counter(tokens)
        
        # Identify unknonw tokens (count = 1)
        for token, count in self.counts.items():
            if count == 1:
                self.unknowns.append(token)
        for i, token in enumerate(tokens):
            if token in self.unknowns:
                tokens[i] = UNK
                
        # calculate prefix counts for n-grams
        prefix_size = self.N-1 
        for i in range(len(tokens)-prefix_size):
            t = tuple(tokens[i:i+prefix_size])
            self.prefix_counts[t] += 1

        # manage start / end tokens, update n-gram counts
        START = '<s>'
        END = '</s>'
        if len(self.unknowns) > 0: 
            self.counts[UNK] = len(self.unknowns)
        
        # UNIGRAM Case: 
        # if n = unigram, don't compute n-gram counts
        if self.N == 1:
            return
        
        for i, word in enumerate(tokens[self.N-1:]):
            i += (self.N-1) # offset by 1
            p = []
            # Get prev word for n-gram counts
            for j in range(i-self.N+1, i):
                p.append(tokens[j])
                if tokens[j] == '<s>':
                    p = ['<s>']
            prev_words = tuple(p)
            # skip sentence boundary tokens when updating n-grams
            if word =='<s>' and '<s>' in prev_words or (word == '<s>' and prev_words[0] == '</s>'):
                continue
            # update n-gram counts for current word & prev words
            self.n_grams_counts[word][prev_words] += 1 # accessing inner dict
            
        # generate n-ngrams and identify sentece start tokens
        for i in range(1,len(tokens)-self.N):
            p = []
            for j in range(i-1, i+self.N-1):
                p.append(tokens[j])
                if tokens[j] == START:
                    p = [START]
                if tokens[j] == END:
                    break
            prev_words = tuple(p)
            
            c = []
            for j in range(i,i+self.N):
                c.append(tokens[j])
                if tokens[j] == START:
                    c = [START]
                if tokens[j] == END:
                    break
            curr_words = tuple(c)
            
            # update generated n-gram counts & find start tokens
            self.generate_n_grams[prev_words][curr_words]+=1
            if curr_words[0] == START:
                self.starts.append(curr_words)
            elif prev_words[0] == START:
                self.starts.append(prev_words)
        
        # remove duplicates 
        self.starts = list(set(self.starts))
        
        # identify knonws (aka any token occuring more than once)
        for each in tokens:
            if each not in self.unknowns:
                self.knowns.add(each)

     
    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model

        Returns:
          float: the probability value of the given tokens for this model
        """
        # STUDENTS IMPLEMENTS
        
        # replace unknowns with <UNK>
        for i,each in enumerate(sentence_tokens):
            if each in self.unknowns:
                sentence_tokens[i] = UNK
        
        # internal function to find prefix count for a given prefix
        def prefix_count(prefix):
            count = 0
            for word in self.n_grams_counts:
                for t in self.n_grams_counts[word]: 
                    # find each occurance of a prefix 
                    if t == prefix: 
                        count += self.n_grams_counts[word][prefix]
            return count
        
        # vocab size
        total_count = len(self.counts) - len(self.unknowns) 

        # UNIGRAM Case: 
        # If n = 1, compute unigram probabilities
        if self.N == 1:
            # replace unk in sentence_tokens
            for i,each in enumerate(sentence_tokens):
                if each not in self.counts:
                    sentence_tokens[i] = UNK
            distinct_words = len(self.counts)
            if self.counts[UNK] > 0:
                distinct_words-=1
            word_count = sum(self.counts.values()) - len(self.unknowns)
            denom = distinct_words + word_count
            prob = 1
            for each in sentence_tokens:
                prob *= ((self.counts[each] + 1) / denom)
            return prob
       
        prob = 1
        prefix = ()
        # N-GRAM Case: n>1 
        for i in range(len(sentence_tokens)):
            if i == (len(sentence_tokens) - 1) :
                break
            word = sentence_tokens[i+1]
            add_to_prefix = sentence_tokens[i]
            
            # creates sliding window 
            if len(prefix) == (self.N-1):
                # remove first word in tuple 
                prefix = prefix[1:] # reassign the tuple to whatever it was 
            prefix = prefix + (add_to_prefix,)
            
            # LAPLACE: Smoothing probability calculated
            numerator = self.n_grams_counts[word][prefix]+ 1 # + 1 is smoothing
            denominator = prefix_count(prefix) + total_count # total count is smoothing
            each_prob = numerator / (denominator) 
            prob *= each_prob
            
        return prob

           
    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          list: the generated sentence as a list of tokens
        """
        # STUDENTS IMPLEMENT
        # imports
        from random import choices, choice
        import copy
        
        # UNIGRAM CASE:
        if self.N == 1:
            res = ['<s>']
            # Create a copy of vocab list without UNK tokens
            counts_cp = copy.copy(self.counts)
            del counts_cp[UNK]
            words, counts = zip(*counts_cp.items())
            
            # Generate sentence until end token is encountered
            while True:
                next_word = choices(words,counts)[0]
                if next_word == '</s>':
                    return res + ['</s>']            
                elif next_word == UNK:
                    next_word = UNK
                elif next_word not in self.knowns:
                    next_word = UNK
                elif next_word == '<s>':
                    continue
                res.append(next_word)
            return res + ['</s>']
        
        # For n-gram models
        next_tuple = choice(self.starts)
        res = list(next_tuple[:-1])
        
        # Generate sentecne until end token is encountered 
        while True:
            if next_tuple != ('<s>',):
                if next_tuple[-1] == UNK:
                    res.append(UNK) #choice(self.unknowns)   #to have words in rather than UNK
                else:
                    res.append(next_tuple[-1])
            tuples, tuple_counts = zip(*self.generate_n_grams[next_tuple].items())
            next_tuple = choices(tuples, tuple_counts)[0]
            if '</s>' in next_tuple:
                break

        return res + ['</s>']
        
    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        # PROVIDED
 
        return [self.generate_sentence() for i in range(n)]


    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
        Args:
          sequence (list): a tokenized sequence to be evaluated for perplexity by this model

        Returns:
          float: the perplexity value of the given sequence for this model
        """
        # 6120 IMPLEMENTS
        # UNIGRAM CASE:
        if self.N == 1:
            denom = sum(self.counts.values())
            prod = 1
            for each in sequence:
                num = self.counts[each]
                each_prod = (num + 1) / (denom + len(self.counts))  
                prod *= each_prod
            # Calculate perplexity as shown in class
            return prod ** (-1 / len(sequence))

        product = 1
        # N-GRAM CASE: n > 1
        for i in range(self.N - 1, len(sequence)):
            prefix = tuple(sequence[i - self.N + 1:i])
            word = sequence[i]

            # Smoothing to numerator & denominator
            numerator = self.n_grams_counts.get(word, {}).get(prefix, 0) + 1
            denom = self.prefix_counts.get(prefix, 0) + len(self.n_grams_counts)  

            each_product = numerator / denom
            product *= each_product
        # calculate perplexity as shown in class
        return product ** (-1 / len(sequence) + 1)

    
# not required
if __name__ == '__main__':
      print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")