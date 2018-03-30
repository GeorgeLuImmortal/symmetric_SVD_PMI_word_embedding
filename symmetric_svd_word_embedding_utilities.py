
# coding: utf-8

# In[1]:


from os import listdir
from os.path import isfile, join
import pandas as pd
import datetime

import preprocessor as p
import string
p.set_options(p.OPT.EMOJI,p.OPT.MENTION,p.OPT.URL)

import csv
from collections import Counter
from nltk.corpus import stopwords

import numpy as np
from itertools import dropwhile
from itertools import chain
import operator

import scipy.sparse as sp
import math

from sklearn.metrics.pairwise import cosine_similarity
import random


# In[3]:


def preprocess_tweet(tweet_list):
    ## remove punctuations, strip punctuations from each token
    ## expect input a list of strings: ["RT @unitetheunion...","Britain needs a ..", ...."proper pay.."...]
    ## expected output a list of seperate tokens [['RT','@unitetheunion'],....['proper','pay']]
    
    
    clean_tweet=[]
    text = [p.clean(i).lower() for i in tweet_list]
    text = [i.split(' ') for i in text]
    
    for tweet in text:
        temp_list=[]
        for token in tweet:
            temp_word = token.strip(string.punctuation)
            temp_list.append(temp_word)
        clean_tweet.append(temp_list)
            
    return clean_tweet

def word_counter(tweet_list):
    ## counter the frequency of token
    ## expected input a list of seperate tokens [['RT','@unitetheunion'//],....['proper','pay'..]]
    ## expected output a python Counter type storing the frequency of each token. e.g counter['hello']:1
    
    counter_ = Counter()
    for tweet in tweet_list:
        counter_.update(tweet)
        
    return counter_

def remove_rare_stop_words(tweet_set,counter,num,stop):
    ## remove stop words and rare terms which frequency less than certain words
    ## expected input a list of tokens [['RT','@unitetheunion'],....['proper','pay']], counter storing individual
    ## words frequency, threshold of the rare term, a list of stop words ['i', 'me', 'my', 'myself',.. , '&amp', 'a', '', 'amp']
    ## expected output a list of seperate tokens without stop words and rare tokens [['@unitetheunion'..],....['proper','pay'..]]
    
    stop_words = stop
    
    rare_terms=[]
    for key in counter:
        if counter[key]<=num:
            rare_terms.append(key)        
            
    ban_list = set(stop_words + rare_terms)
    
    clean_tweet_remove_stop = []
    for tweet in tweet_set:
        temp = []
        [temp.append(term) for term in tweet if term not in ban_list]
        clean_tweet_remove_stop.append(temp)
    
    return clean_tweet_remove_stop

def collocation_counter(tweet_list,order=False):
    ## count the co-occurrences of any two words
    ## expected input a list of seperated cleaned tokens [['@unitetheunion'..],....['proper','pay'..]]
    ## a order flag, default is False, that means ignoring the order, it will treat ['a','b'] and ['b','a'] the same
    ## expected output a counter storing the cooccurrences of bigrams e.g. counter[('hellow','world)]:17
    

    if order == True:
        count_collocation = Counter()
        for tweet in tweet_list:
            for idx,word in enumerate(tweet[:-1]):
                for i in range(idx,len(tweet)-1):
                    count_collocation.update(((word,tweet[i+1]),))



        return count_collocation
    else:
        count_collocation = Counter()
        for tweet in tweet_list:
            for idx,word in enumerate(tweet[:-1]):
                for i in range(idx,len(tweet)-1):
                    if (tweet[i+1],word) in count_collocation:
                        count_collocation.update(((tweet[i+1],word),))
                    else:
                        count_collocation.update(((word,tweet[i+1]),))


        return count_collocation

def remove_rare_term(counter_,num):
    ## remove the low frequency bigrams from the bigram counter
    ## expected input a counter of bigrams, number of frequency threshold
    ## expected output a counter storing the bigrams with frequency higher than threshold, e.g. counter[('hellow','world)]:17
    
    counter = counter_.copy()
    for key, count in dropwhile(lambda key_count: key_count[1] > num, counter.most_common()):
        del counter[key]
        
    return counter
  

    
def construct_word_occurrence_matrix(counter_uni,counter_col):
    
    ## construct a word co-occurrence matrix
    ## expected input, a counter storing uni-gram frequency, a counter storing bigram frequency
    ## expected output, a pandas dataframe of co-occurences, and a np array values, shape is the (len(vocab),len(vocab))

    
    vocab_size = len(counter_uni)
    header = counter_uni.keys()
    a = np.zeros((vocab_size,vocab_size))
    frame = pd.DataFrame(a,index=header,columns=header)
    
    for item in counter_col:
        frame[item[0]][item[1]] = counter_col[item]
        frame[item[1]][item[0]] = counter_col[item]
        
    return frame,frame.values
    
    
def PPMI_matrix(names_index,names_header,matrix,la_smooth=False):
    
    ## construct a positive-pmi matrix
    ## expected input, the index, header, usually is a list of vocabulary ['allen','apple','banana',..'zoo']
    ## the word co-occurrence matrix, laplace smooth value, default is false, expected smooth range from 0-3
    ## expected output a PPMI matrix
    
    if la_smooth:
        print('applying laplace smoothed, parameter is', la_smooth)
        
        matrix_sm = matrix.copy()
        matrix_sm = matrix_sm + la_smooth
        N = matrix_sm.sum()
        
    
        num_row = matrix_sm.shape[0]
        num_col = matrix_sm.shape[1]

        col = matrix_sm.sum(0)
        row = matrix_sm.sum(1)

        row = row.reshape(num_row,1)
        col = col.reshape(1,num_col)

        denominator_matrix = np.dot(row,col)

        pmi = np.log2(matrix_sm*N/denominator_matrix)

        pmi[np.isnan(pmi)]=0
        pmi[pmi<0]=0

        dff = pd.DataFrame(pmi,index=names_index,columns=names_header)
        
    else:
        print('no smooth applied')
        N = matrix.sum()

        num_row = matrix.shape[0]
        num_col = matrix.shape[1]

        col = matrix.sum(0)
        row = matrix.sum(1)

        row = row.reshape(num_row,1)
        col = col.reshape(1,num_col)

        denominator_matrix = np.dot(row,col)

        pmi = np.log2(matrix*N/denominator_matrix)

        pmi[np.isnan(pmi)]=0
        pmi[pmi<0]=0

        dff = pd.DataFrame(pmi,index=names_index,columns=names_header)
    
    return dff,pmi
    

def construct_word_embedding(d,matrix):
    
    ## use SVD factorization a PMI matrix
    ## expected input, the dimenstion of output matrix, the PPMI matrix
    ## expected output, the symmetrics matrix W, C , shape is (len(vocab),d)
    
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    
    sigma = np.zeros((d,d))
    sigma = np.diag(s[:d])
    sigma = np.sqrt(sigma)
    
    W = np.dot(u[:,:d],sigma)
    C = np.dot(vh[:,:d],sigma)
    
    return W,C
    
    
def to_csv(matrix,header,filename):
    
    ## export the word embedding dataframe to a csv file and return the dataframe
    ## expected input, the embedding matrix, the index of dataframe, the filename for outputint
    
    frame = pd.DataFrame(matrix,index=header)
    frame.to_csv(filename)
    
    return frame
    
def read_from_csv(mypath,header_row):
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print('the filenames are ***************************')
    print(onlyfiles)
    
    df_raw = pd.DataFrame(columns=header_row)
    for idx,filename in enumerate(onlyfiles):
        df = pd.read_csv(mypath+'/'+filename,names=header_row)
        frames = [df_raw,df]
        df_raw = pd.concat(frames)
        
    print('Original size of data is ', len(df_raw))
    df_raw = df_raw.drop_duplicates(subset='text')
    df_raw = df_raw.reset_index(drop=True)
    print('after remove duplicate, the size is ',len(df_raw))
    print("\n")
    
    return df_raw
    
def read_from_txt(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print('the filenames are ***************************')
    print(onlyfiles)
    
    df_raw = pd.DataFrame()
    for idx,filename in enumerate(onlyfiles):
        df = pd.read_csv(mypath+'/'+filename,sep=" ")
        frames = [df_raw,df]
        df_raw = pd.concat(frames)
        
    print('Original size of data is ', len(df_raw))
    df_raw = df_raw.drop_duplicates(subset='text')
    df_raw = df_raw.reset_index(drop=True)
    print('after remove duplicate, the size is ',len(df_raw))
    print("\n")

       
    return df_raw

def co_occurrence_matrix(document):
    
    ## another way for constructing co_occurrence matrix, P.S. this method requiring large memory 
    ## expected input a list of seperated cleaned tokens [['@unitetheunion'..],....['proper','pay'..]]
    ## expected output the word co-occurrences matrix
    
    names = sorted(set(list(chain(*document))))
    voc2id = dict(zip(names, range(len(names))))
    rows, cols, vals = [], [], []
    for r, d in enumerate(document):
        for e in d:
            if voc2id.get(e) is not None:
                rows.append(r)
                cols.append(voc2id[e])
                vals.append(1)
    X = sp.csr_matrix((vals, (rows, cols)))
    
    Xc = (X.T * X) # coocurrence matrix
    Xc.setdiag(0)
    a = Xc.toarray()
    
    dff = pd.DataFrame(a,index = names, columns=names)
    
    return dff,a

def show_most_similar(word,df,num):
    
    cos_sim_matrix = cosine_similarity(df)
    
    frame1=df.reset_index(drop=False)
    
    target_index=frame1[frame1['index'] == word].index.values[0]
    
    arr = np.array(cos_sim_matrix[target_index])

    order = arr.argsort()[-100:][::-1]


    for i in order:
        print(frame1.loc[i]['index'])
        
    return order


def remove_high_frequency_words(tweet_set,counter_uni,except_list,t):
    ## remove stop words and rare terms which frequency less than certain words
    ## expected input a list of tokens [['RT','@unitetheunion'],....['proper','pay']], counter storing individual
    ## words frequency, threshold of the rare term, a list of stop words ['i', 'me', 'my', 'myself',.. , '&amp', 'a', '', 'amp']
    ## expected output a list of seperate tokens without stop words and rare tokens [['@unitetheunion'..],....['proper','pay'..]]
    
    lent = sum(counter_uni.values())
    counter_drop = {}
    for key in counter_uni:
        f = counter_uni[key]/lent
        counter_drop[key] = ((f-t)/f)-math.sqrt((t/f))
        
        
    remove_list=[]
    for key in counter_drop:
        p = random.random()
        if p < counter_drop[key]:
            remove_list.append(key)
        
    remove_list_final=[item for item in remove_list if item not in except_list]
    print(remove_list_final)
         
    
    clean_tweet_remove_stop = []
    for tweet in tweet_set:
        temp = []
        [temp.append(term) for term in tweet if term not in remove_list_final]
        clean_tweet_remove_stop.append(temp)
    
    return clean_tweet_remove_stop

