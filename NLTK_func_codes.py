# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script that contains the learning and funcitons from NLTK book in 
Python. Comments have also been added

"""

from __future__ import division

'''
laod all the books from python

all the books are loaded as type :nltk.text.Text
if you are creating a custom book, please confirm that it is of the same class
for all the other funcitons to work

'''
import nltk
from nltk.book import *



'''
searching some text in a book
'''

text1.concordance("monstrous")


'''
finding similar words in a context
'''

text1.similar("monstrous")


'''
context that is shared by two or more words
'''

text1.common_contexts(["monstrous","very"])


'''
location of a word in a text
'''

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
 


'''
COUNTING VOCABULARY
'''


text3.count("a") 	
len(text3)   #total words in a text
sorted(set(text3)) #unique words in sorted order
len(set(text3)) #number of unique words in a text


len(set(text3))/len(text3)   #measure of lexical richness of a text

#funtion to retuen lexical richness
def lexical_diversity(text):
    return len(set(text))/len(text)
    
#function to return word count in percentage
def word_precentage(word,text):
    return 100*text.count(word)/len(text)
    
    
'''
Converting a normal string into nltk.text.Text format
'''

str1 = "Hello! How are you"
toks = nltk.word_tokenize(str1)
custom_book = nltk.text.Text(toks)

    
'''
Frequency Distribution
'''

freqD = FreqDist(text1) #list of all the words along with the Frequency
freqTop = freqD.most_common(50)  #extracting top 50 words


sorted(w for w in set(text1) if len(w) > 7 and freqD[w] > 7)#soritng and 
#filtering using text size and frequency    



'''
Collocations and Bigrams
'''

list(nltk.bigrams(['more', 'is', 'said', 'than', 'done']))
#makes all the bigrams from the list of words

coloc = text1.collocations()


'''
couting other things
'''

[len(w) for w in text1]  #calcluates length of each word
fdist = FreqDist(len(w) for w in text1)  #counts the number of words for each length
fdist.most_common() #filters top 50 most frequent word length
fdist.max()  #gives the maximum
fdist.freq(3) #gives the frequency share

fdist = FreqDist(samples)	#create a frequency distribution containing the given samples
fdist[sample] += 1	#increment the count for this sample
fdist['monstrous']	#count of the number of times a given sample occurred
fdist.freq('monstrous')	#frequency of a given sample
fdist.N()	#total number of samples
fdist.most_common(n)	#the n most common samples and their frequencies
fdist.max()	#sample with the greatest count
fdist.tabulate()	#tabulate the frequency distribution
fdist.plot()	#graphical plot of the frequency distribution
fdist.plot(cumulative=True)	#cumulative plot of the frequency distribution
fdist1 |= fdist2	#update fdist1 with counts from fdist2
fdist1 < fdist2	#test if samples in fdist1 occur less frequently than in fdist2



    
'''
CH-2
Accessing Text Corpora and Lexical Resources

Practical work in Natural Language Processing typically uses large bodies of 
linguistic data, or corpora. 
'''


'''
Gutenberg Corpus

NLTK includes a small selection of texts from the Project Gutenberg electronic 
text archive, which contains some 25,000 free electronic books, hosted at 
http://www.gutenberg.org/. We begin by getting the Python interpreter to load
the NLTK package, then ask to see nltk.corpus.gutenberg.fileids(), the file 
identifiers in this corpus
'''

import nltk
nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.words('austen-emma.txt')

emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
 
emma.concordance("surprize")   
    
from nltk.corpus import gutenberg

'''
writing a short program to calculate a few statistics for each books in 
gutenberg books list
'''
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid)) [1]
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
    


macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')    

macbeth_sentences

'''
finding and priting the longest sentence in a corpus
'''

longest_len = max(len(s) for s in macbeth_sentences)

[s for s in macbeth_sentences if len(s) == longest_len]
 
'''
brown corpus after gutenberg corpus  : more vivid and has categories
'''
 
from nltk.corpus import brown 

brown.categories()


brown.words(categories='news')

brown.words(fileids=['cg22'])

brown.sents(categories=['news', 'editorial', 'reviews'])

'''
web and chat transcript
'''
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')


'''
The Brown Corpus is a convenient resource for studying systematic differences between genres, a kind of linguistic 
inquiry known as stylistics. Let's compare genres in their usage of modal verbs. 
'''

 	
from nltk.corpus import brown
news_text = brown.words(categories='news') 
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print " ".join((str(m),":",str(fdist[m])))
    
    
cfd = nltk.ConditionalFreqDist(
          (genre, word)
          for genre in brown.categories()
          for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)    


'''
Reuters Corpus
The Reuters Corpus contains 10,788 news documents totaling 
1.3 million words. The documents have been classified into 90 topics, 
and grouped into two sets, called "training" and "test"; thus, the text 
with fileid 'test/14826' is a document drawn from the test set.
'''

from nltk.corpus import reuters
reuters.fileids()
reuters.categories()

reuters.categories('training/9865')  #finding the category of a fileid
reuters.fileids('barley') #barley is a category

'''
 Inaugural Address Corpus
 
'''

from nltk.corpus import inaugural
inaugural.fileids()


cfd = nltk.ConditionalFreqDist(
      (target, fileid[:4])
      for fileid in inaugural.fileids()
      for w in inaugural.words(fileid)
      for target in ['america', 'citizen']
      if w.lower().startswith(target))
cfd.plot()


'''
Annotated Text Corpora
Many text corpora contain linguistic annotations, representing POS tags, 
named entities, syntactic structures, semantic roles, and so forth. 
NLTK provides convenient ways to access several of these corpora, and 
has data packages containing corpora and corpus samples, freely 
downloadable for use in teaching and research.

Examples: Brown
'''

nltk.corpus.fileids()	#the files of the corpus
nltk.corpus.fileids([categories])	#the files of the corpus corresponding to these categories
nltk.corpus.categories()	#the categories of the corpus
nltk.corpus.categories([fileids])	#the categories of the corpus corresponding to these files
nltk.corpus.raw()	#the raw content of the corpus
nltk.corpus.raw(fileids=[f1,f2,f3])	#the raw content of the specified files
nltk.corpus.raw(categories=[c1,c2])	#the raw content of the specified categories
nltk.corpus.words()	#the words of the whole corpus
nltk.corpus.words(fileids=[f1,f2,f3])	#the words of the specified fileids
nltk.corpus.words(categories=[c1,c2])	#the words of the specified categories
nltk.corpus.sents()	#the sentences of the whole corpus
nltk.corpus.sents(fileids=[f1,f2,f3])	#the sentences of the specified fileids
nltk.corpus.sents(categories=[c1,c2])	#the sentences of the specified categories
nltk.corpus.abspath(fileid)	#the location of the given file on disk
nltk.corpus.encoding(fileid)	#the encoding of the file (if known)
#open(fileid)	#open a stream for reading the given corpus file
nltk.corpus.root	#if the path to the root of locally installed corpus
nltk.corpus.readme()	#the contents of the README file of the corpus



'''
Loading your own corpus
'''

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('connectives')


'''
Loading a local copy of any of the NLTK corpus
'''

from nltk.corpus import BracketParseCorpusReader
corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj" [1]
file_pattern = r".*/wsj_.*\.mrg" [2]
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()
#['00/wsj_0001.mrg', '00/wsj_0002.mrg', '00/wsj_0003.mrg', '00/wsj_0004.mrg', ...]
len(ptb.sents())
ptb.sents(fileids='20/wsj_2013.mrg')[19]



'''
Conditional Frequency Distributions
It is just a frequency count on a category level
'''
'''
Conditions and Events
'''
text = ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said']
pairs = [('news', 'The'), ('news', 'Fulton'), ('news', 'County')]
         
'''
Counting Words by Genre
''' 

from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))        
         
         
genre_word = [(genre, word) [1]
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)]

cfd=nltk.ConditionalFreqDist(genre_word)



'''
Plotting and Tabulating Distributions
'''

