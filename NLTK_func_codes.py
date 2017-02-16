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

Collocations are frequent bigrams
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

from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))



from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word)) [1]
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))


cfd.tabulate(conditions=['English', 'German_Deutsch'],
       samples=range(10), cumulative=True)



'''
Generating Random Text with Bigrams
'''

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
'and', 'the', 'earth', '.']
list(nltk.bigrams(sent))


def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word)
        word = cfdist[word].max()
        
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)        
        
cfd['living']
generate_model(cfd, 'living')



'''
CondtionalFreqDist functions and properties
'''
from nltk import ConditionalFreqDist
condition=['living']
conditions=['living']
sample=5
samples=5
cfdist = ConditionalFreqDist(bigrams)	#create a conditional frequency distribution from a list of pairs
cfdist.conditions()	#the conditions
cfdist[condition]	#the frequency distribution for this condition
cfdist[condition][sample]	#frequency for the given sample for this condition
cfdist.tabulate()	#tabulate the conditional frequency distribution
cfdist.tabulate(samples, conditions)	#tabulation limited to the specified samples and conditions
cfdist.plot()	#graphical plot of the conditional frequency distribution
cfdist.plot(samples, conditions)	#graphical plot limited to the specified samples and conditions
cfdist1 < cfdist2	#test if samples in cfdist1 occur less frequently than in cfdist2


'''
writing reusable functions
'''



from __future__ import division
def lexical_diversity(text):
    return len(text) / len(set(text))
    
def lexical_diversity(my_text_data):
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = vocab_size / word_count
    return diversity_score    
    
'''
function to retuen the plural of a word
'''
    
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'    
    
plural('fairy')
plural('woman')



'''
Lexical Resources


A lexicon, or lexical resource, is a collection of words and/or phrases 
along with associated information such as part of speech and sense 
definitions. Lexical resources are secondary to texts, and are usually 
created and enriched with the help of texts.        
'''

'''
A lexical entry consists of a headword (also known as a lemma) along 
with additional information such as the part of speech and the sense 
definition. Two distinct words having the same spelling are called homonyms.
'''

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)
    
unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))    



'''
stopwords
'''


from nltk.corpus import stopwords

stopwords.words("english")

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
    
content_fraction(nltk.corpus.reuters.words())


puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6
and obligatory in w 
and nltk.FreqDist(w) <= puzzle_letters]




'''
names of men and women
'''

names = nltk.corpus.names
names.fileids()
names.words(names.fileids()[0])  #women


cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1])
    for fileid in names.fileids()
    for name in names.words(fileid))
cfd.plot()


'''
A Pronouncing Dictionary
'''

entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[42371:42379]:
    print(entry)

'''
('fir', ['F', 'ER1'])
('fire', ['F', 'AY1', 'ER0'])
('fire', ['F', 'AY1', 'R'])
('firearm', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M'])
('firearm', ['F', 'AY1', 'R', 'AA2', 'R', 'M'])
('firearms', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M', 'Z'])
('firearms', ['F', 'AY1', 'R', 'AA2', 'R', 'M', 'Z'])
('fireball', ['F', 'AY1', 'ER0', 'B', 'AO2', 'L'])
'''


 	
for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word," ",ph2)
            
            
'''
syllable example
'''


syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]



'''
The phones contain digits to represent primary stress (1), 
secondary stress (2) and no stress (0). As our final example, 
we define a function to extract the stress digits and then scan \
our lexicon to find words having a particular stress pattern.
'''

def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]
[w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]  


[w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']] 


'''
finding minimally contrasting words
'''

p3 = [(pron[0]+'-'+pron[2], word)
    for (word, pron) in entries
        if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
    if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring[:70] + "...")
        

'''
making dictionary
'''        
prondict = nltk.corpus.cmudict.dict()    
prondict['fire']
prondict['blog']  #gives error

prondict['blog'] = [['B', 'L', 'AA1', 'G']]

text = ['natural', 'language', 'processing']
[ph for w in text for ph in prondict[w][0]]


'''
Comparative Wordlists
Another example of a tabular lexicon is the comparative wordlist. 
NLTK includes so-called Swadesh wordlists, lists of about 200 common 
words in several languages. 
The languages are identified using an ISO 639 two-letter code.
'''

from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')


fr2en = swadesh.entries(['fr', 'en'])
fr2en
translate = dict(fr2en)
translate['chien']     


'''
Shoebox and Toolbox Lexicons
A Toolbox file consists of a collection of entries, where each entry is made
 up of one or more fields. Most fields are optional or repeatable, which 
 means that this kind of lexical resource cannot be treated as a table or 
 spreadsheet.
'''
from nltk.corpus import toolbox
toolbox.entries('rotokas.dic')


'''
Entries consist of a series of attribute-value pairs, like ('ps', 'V') 
to indicate that the part-of-speech is 'V' (verb), and ('ge', 'gag') 
to indicate that the gloss-into-English is 'gag'. The last three pairs 
contain an example sentence in Rotokas and its translations into Tok 
Pisin and English.
'''


'''
WORDNET

WordNet is a semantically-oriented dictionary of English, similar to a 
traditional thesaurus but with a richer structure. NLTK includes the English
WordNet, with 155,287 words and 117,659 synonym sets. We'll begin by 
looking at synonyms and how they are accessed in WordNet.
'''

from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').definition() 
wn.synset('car.n.01').examples()
wn.synset('car.n.01').lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()

#another example
wn.synsets('car')
for synset in wn.synsets('car'):
    print(synset.lemma_names())
    
wn.lemmas('car')    
    

'''
Wordnet Heirarchy
'''

'''
WordNet synsets correspond to abstract concepts, and they don't always 
have corresponding words in English. These concepts are linked together 
in a hierarchy. Some concepts are very general, such as Entity, State, 
Event — these are called unique beginners or root synsets.
'''


motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0]
sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())


motorcar.hypernyms()
paths = motorcar.hypernym_paths()
[synset.name() for synset in paths[0]]
 
motorcar.root_hypernyms()


'''
More Lexical Relations

Hypernyms and hyponyms are called lexical relations because they relate 
one synset to another. These two relations navigate up and down the "is-a" 
hierarchy. Another important way to navigate the WordNet network is from 
items to their components (meronyms) or to the things they are contained in 
(holonyms). For example, the parts of a tree are its trunk, crown, and so 
on; the  part_meronyms(). The substance a tree is made of includes heartwood 
and sapwood; the substance_meronyms(). A collection of trees forms a forest;
the member_holonyms():
'''
wn.synset('tree.n.01').part_meronyms()
wn.synset('tree.n.01').substance_meronyms()
wn.synset('tree.n.01').member_holonyms()

for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())
    
wn.synset('mint.n.04').part_holonyms()
wn.synset('mint.n.04').substance_holonyms()

'''
There are also relationships between verbs. For example, the act of 
walking involves the act of stepping, so walking entails stepping. 
Some verbs have multiple entailments    
'''

wn.synset('walk.v.01').entailments()
wn.synset('eat.v.01').entailments()
wn.synset('tease.v.03').entailments()


'''
Some lexical relationships hold between lemmas, e.g., antonymy:
'''
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms()
wn.lemma('horizontal.a.01.horizontal').antonyms()


'''
Semantic
'''

 	
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
right.lowest_common_hypernyms(minke)
[Synset('baleen_whale.n.01')]
right.lowest_common_hypernyms(orca)
[Synset('whale.n.02')]
right.lowest_common_hypernyms(tortoise)
[Synset('vertebrate.n.01')]
right.lowest_common_hypernyms(novel)
[Synset('entity.n.01')]
 
 
wn.synset('baleen_whale.n.01').min_depth()
wn.synset('whale.n.02').min_depth()
wn.synset('vertebrate.n.01').min_depth()
wn.synset('entity.n.01').min_depth()
 

 	
right.path_similarity(minke)
right.path_similarity(orca)
right.path_similarity(tortoise)
right.path_similarity(novel)



'''
CHAPTER 3: PROCESSING RAW DATA
'''

from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize


'''
reading materials from web
'''
    
import urllib2
url = "http://www.gutenberg.org/files/2554/2554.txt"
response = urllib2.urlopen(url)
raw = response.read().decode('utf8')
type(raw)

tokens = word_tokenize(raw)

text = nltk.Text(tokens)
text = nltk.text.Text(tokens)

'''
The find() and rfind() ("reverse find") methods help us get the right 
index values to use for slicing the string [1]. We overwrite raw with 
this slice, so now it begins with "PART I" and goes up to 
(but not including) the phrase that marks the end of the content.
'''

raw.find("PART I")
raw.rfind("PART I")



'''
Dealing with HTML
'''


url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
response = urllib2.urlopen(url)
html = response.read().decode('utf8')

 	
from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens

'''
concordance helps us to locate the word in the text and returns it as a
text
'''

tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance('gene')


'''
Processing Search Engine Results

Your Turn: Search the web for "the of" (inside quotes). 
Based on the large count, can we conclude that the of is a 
frequent collocation in English?
'''

'''
Processing RSS Feeds
'''

 	
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']

len(llog.entries)
post = llog.entries[2]
post.title
content = post.content[0].value
raw = BeautifulSoup(content).get_text()
word_tokenize(raw)



'''
Reading Local Files
'''
f = open('/home/saurabh/Desktop/installations.txt')
raw = f.read()
#to read each line please use readline()

'''
Extracting Text from PDF, MSWord and other Binary Formats, specialized 
softwares are available
'''
s = input("Enter some text: ")
print("You typed", len(word_tokenize(s)), "words.")



'''
The NLP Pipeline

HTML > ASCII > TEXT > VOCAB
'''

'''
Simple text operations
'''

couplet = '''Rough winds do shake the darling buds of May,
... And Summer's lease hath all too short a date:'''

couplet = "Shall I compare thee to a Summer's day?"\
          "Thou are more lovely and more temperate:"
          
adding_words = 'very' + 'very' + 'very'  

multiplying_text = 'very' * 3         

 	
a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b:
    print(line)
    
    
'''
finding most used character in a corpus
'''    

from nltk.corpus import gutenberg
raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
fdist.most_common(5)


'''
More operations on strings
'''

s="i am the best in the world"
t="best"
u="worst"

s.find(t)	     #index of first instance of string t inside s (-1 if not found)
s.rfind(t)	#index of last instance of string t inside s (-1 if not found)
s.index(t)	#like s.find(t) except it raises ValueError if not found
s.rindex(t)	#like s.rfind(t) except it raises ValueError if not found
s.join(text)	#combine the words of the text into a string using s as the glue
s.split(t)	#split s into a list wherever a t is found (whitespace by default)
s.splitlines()	#split s into a list of strings, one per line
s.lower()  	#a lowercased version of the string s
s.upper()	     #an uppercased version of the string s
s.title()	     #a titlecased version of the string s
s.strip()	     #a copy of s without leading or trailing whitespace
s.replace(t, u)	#replace instances of t with u inside s

'''
The concept of "plain text" is a fiction. 

In this section, we will give an overview of how to use Unicode for processing 
texts that use non-ASCII character sets.

Unicode supports over a million characters. Each character is assigned a 
number, called a code point. In Python, code points are written in the form 
\uXXXX, where XXXX is the number in 4-digit hexadecimal form.



However, when Unicode characters are stored in files or displayed on a 
terminal, they must be encoded as a stream of bytes. Some encodings 
(such as ASCII and Latin-2) use a single byte per code point, so they can only 
support a small subset of Unicode, enough for a single language. Other 
encodings (such as UTF-8) use multiple bytes and can represent the full range 
of Unicode characters.

From a Unicode perspective, characters are abstract entities which can be 
realized as one or more glyphs. Only glyphs can appear on a screen or be 
printed on paper. A font is a mapping from characters to glyphs.
'''

#path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')

#f = open(path, encoding='latin2')
#for line in f:
#    line = line.strip()
#    print(line)



'''
Regular Expressions for Detecting Word Patterns
'''

import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

'''
Let's find words ending with ed using the regular expression «ed$».
'''
            
[w for w in wordlist if re.search('ed$', w)]

'''
The . wildcard symbol matches any single character. 
Suppose we have room in a crossword puzzle for an 8-letter word 
with j as its third letter and t as its sixth letter. In place of each 
blank cell we use a period:
'''
 
[w for w in wordlist if re.search('^..j..t..$', w)]
 

'''
The first part of the expression, «^[ghi]», matches the start of a word 
followed by g, h, or i. The next part of the expression, «[mno]», 
constrains the second character to be m, n, or o.
''' 

[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)] 
 
'''
Let's explore the + symbol a bit further. Notice that it can be applied to 
individual letters, or to bracketed sets of letters:
'''    

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]
[w for w in chat_words if re.search('^[ha]+$', w)]


wsj = sorted(set(nltk.corpus.treebank.words())) 

[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]
[w for w in wsj if re.search('^[A-Z]+\$$', w)]
 
'''
will search for numbers 4 times till end of string
''' 
[w for w in wsj if re.search('^[0-9]{4}$', w)] 
 
[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]
 
[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]

[w for w in wsj if re.search('(ed|ing)$', w)] 
 
 
 
'''
Basic Regular Expression Meta-Characters, Including Wildcards, Ranges and 
Closures
'''
'''
^abc	Matches some pattern abc at the start of a string
abc$	Matches some pattern abc at the end of a string
[abc]	Matches one of a set of characters
[A-Z0-9]	Matches one of a range of characters
ed|ing|s	Matches one of the specified strings (disjunction)
*	Zero or more of previous item, e.g. a*, [a-z]* (also known as Kleene Closure)
+	One or more of previous item, e.g. a+, [a-z]+
?	Zero or one of the previous item (i.e. optional), e.g. a?, [a-z]?
{n}	Exactly n repeats where n is a non-negative integer
{n,}	At least n repeats
{,n}	No more than n repeats
{m,n}	At least m and no more than n repeats
a(b|c)+	Parentheses that indicate the scope of the operators 
''' 

'''
Useful Applications of Regular Expressions
'''

'''
Extracting Word Pieces
'''

word = 'supercalifragilisticexpialidocious'
re.findall(r'[aeiou]', word)
len(re.findall(r'[aeiou]', word))


wsj = sorted(set(nltk.corpus.treebank.words()))


fd = nltk.FreqDist(vs for word in wsj
for vs in re.findall(r'[aeiou]{2,}', word))


'''
Doing More with Word Pieces
'''

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)
    
english_udhr = nltk.corpus.udhr.words('English-Latin1')    
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))


'''
combining conditional frequency with regex
'''
rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()       


cv_word_pairs = [(cv, w) for w in rotokas_words 
                 for cv in re.findall(r'[ptksvr][aeiou]', w)]

cv_index = nltk.Index(cv_word_pairs)
cv_index['su']
cv_index['po']



'''
making a stemmer
'''

re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')
re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')
re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language')


def stem(word):
   
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem
    
    
raw = """DENNIS: Listen, strange women lying in ponds distributing swords 
is no basis for a system of government.  Supreme executive power derives from 
a mandate from the masses, not from some farcical aquatic ceremony."""    
    
tokens = nltk.word_tokenize(raw)
[stem(t1) for t1 in tokens]
 
 
'''
Searching Tokenized Text
'''

'''
examples to mark different word boundaries
'''

from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a> (<.*>) <man>")
moby.concordance("monied")

chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*> <.*> <bro>")
chat.findall(r"<l.*>{3,}")


from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")



'''
Normalizing Texts
'''

'''
Stemmers : Porter and lancaster
'''

raw = """DENNIS: Listen, strange women lying in ponds distributing swords 
is no basis for a system of government.  Supreme executive power derives from
 a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = nltk.word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]
 
 
 
'''
Object Oriented Python

Writing a class to identify concordance sentences and perform stemming 
''' 	
class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower() 
        
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')
        

'''
Lemmatization

The WordNet lemmatizer only removes affixes if the resulting word is in 
its dictionary. This additional checking process makes the lemmatizer 
slower than the above stemmers. Notice that it doesn't handle lying, 
but it converts women to woman.
The WordNet lemmatizer is a good choice if you want to compile the 
vocabulary of some texts and want a list of valid lemmas (or lexicon headwords).

'''

wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]
 
 
 
'''
Regular Expressions for Tokenizing Text
'''

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
 though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
 well without--Maybe it's always pepper that makes people hot-tempered,'""" 
 
 
 
re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)

re.split(r'\W+', raw) # \w means [a-zA-Z0-9_]

re.findall(r'\w+|\S\w*', raw)
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))


'''
Regular Expression Symbols

Symbol	Function
\b	Word boundary (zero width)
\d	Any decimal digit (equivalent to [0-9])
\D	Any non-digit character (equivalent to [^0-9])
\s	Any whitespace character (equivalent to [ \t\n\r\f\v])
\S	Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
\w 	Any alphanumeric character (equivalent to [a-zA-Z0-9_])
\W 	Any non-alphanumeric character (equivalent to  [^a-zA-Z0-9_])
\t	The tab character
\n	The newline character

''' 

'''
NLTK's Regular Expression Tokenizer
'''

text = 'That U.S.A. poster-print costs $12.40...'

pattern = r'''(?x)    # set flag to allow verbose regexps
...     ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
...   | \w+(-\w+)*        # words with optional internal hyphens
...   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
...   | \.\.\.            # ellipsis
...   | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
... '''
nltk.regexp_tokenize(text, pattern)
['That', 'U.S.A.', 'poster-print', 'costs', '$12.40', '...']




'''
SEGMENTATION

Tokenization is an instance of a more general problem of segmentation. 
In this section we will look at two other instances of this problem, 
which use radically different techniques to the ones we have seen so far 
in this chapter.
'''


'''
Sentence Segmentation
'''

#average number of words per sentence

len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())



text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
pprint.pprint(sents[79:89])


'''
Our first challenge is simply to represent the problem: we need to find a 
way to separate text content from the segmentation. We can do this by 
annotating each character with a boolean value to indicate whether or 
not a word-break appears after the character (an idea that will be used 
heavily for "chunking" in 7.). Let's assume that the learner is given the 
utterance breaks, since these often correspond to extended pauses. Here is 
a possible representation, including the initial and target segmentations:
    '''

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"

def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words
    
segment(text, seg1)
segment(text, seg2)    


'''
Now the segmentation task becomes a search problem: find the bit string 
that causes the text string to be correctly segmented into words
'''

'''
we can define an objective function, a scoring function whose value we will 
try to optimize, based on the size of the lexicon (number of characters in 
the words plus an extra delimiter character to mark the end of each word) 
and the amount of information needed to reconstruct the source text from 
the lexicon.
'''

    
    
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"



'''
writing a code to perform simulated annealing to perform word segmentation
'''

def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(word) + 1 for word in set(words))
    return text_size + lexicon_size


from random import randint

def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs
    
    
    
'''
WRITING STRUCTURED PROGRAM IN PYTHON
'''    

size = 5
python = ['Python']
snake_nest = [python] * size
snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]
True
snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]
True

import random
position = random.choice(range(size))
snake_nest[position] = ['Python']
snake_nest

snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]
True
snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]
True

#knowing the stored location and object id
[id(snake) for snake in snake_nest]
 
 
#iterators
'''
for item in s	iterate over the items of s
for item in sorted(s)	iterate over the items of s in order
for item in set(s)	iterate over unique elements of s
for item in reversed(s)	iterate over elements of s in reverse
for item in set(s).difference(t)	iterate over elements of s not in t 
'''

'''
making the train and test data
'''

 	
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
True
len(training_data) / len(test_data)




words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
wordlens.sort()
' '.join(w for (_, w) in wordlens)


text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone,
... "it means just what I choose it to mean - neither more nor less."'''

[w.lower() for w in nltk.word_tokenize(text)]
 
#finds the words which comes latest in lexographic sort order 
max([w.lower() for w in nltk.word_tokenize(text)])



'''
cumulative freq dist
'''

fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
       break
   
   
   
'''
finding the longest word in a corpus
'''

text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
    if len(word) > len(longest):
        longest = word
longest


maxlen = max(len(word) for word in text)
[word for word in text if len(word) == maxlen]
 

'''
Functions
'''
 
import re
def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file).read()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text
    
    
'''
call by value and reference example
'''

def set_up(word, properties):
    word = 'lolcat'
    properties.append('noun')
    properties = 5    
 
w=''
p=[]    

set_up(w,p)
    

'''
Variable Scope

Function definitions create a new, local scope for variables. When you 
assign to a new variable inside the body of a function, the name is only 
defined within that function. The name is not visible outside the function, 
or in other functions. This behavior means you can choose variable names 
without being concerned about collisions with names used in your other 
function definitions.

When you refer to an existing name from within the body of a function, 
the Python interpreter first tries to resolve the name with respect to the 
names that are local to the function. If nothing is found, the interpreter 
checks if it is a global name within the module. Finally, if that does not 
succeed, the interpreter checks if the name is a Python built-in. This is 
the so-called LGB rule of name resolution: local, then global, then built-in.
'''


'''
function to check input and return assertion/type error
'''

 	
def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'
        
        
    
'''
Function documentation
'''        

def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.

    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.

        >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
        0.5

    :param reference: An ordered list of reference values
    :type reference: list
    :param test: A list of values to compare against the corresponding
        reference values
    :type test: list
    :return: the accuracy score
    :rtype: float
    :raises ValueError: If reference and length do not have the same length
    """

    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in zip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)
    
    
'''
Doing more with functions
'''

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
       'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
def extract_property(prop):
    return [prop(word) for word in sent]
            
extract_property(len)
def last_letter(word):
    return word[-1]
extract_property(last_letter)   

extract_property(lambda w: w[-1])    




'''
Accumulative Functions : Functions collects the result before returning
'''

def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result

def search2(substring, words):
    for word in words:
        if substring in word:
            yield word     
            
for item in search1('zz', nltk.corpus.brown.words()):
   print item            
   
for item in search2('zz', nltk.corpus.brown.words()):
   print item 
   
#making a function to do permutations of a given list items
def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + seq[0:1] + perm[i:]

list(permutations(['police', 'fish', 'buffalo'])) 


'''
named and keyword arguments
'''

def generic(*args, **kwargs):
    print(args)
    print(kwargs)
    
    
generic(1,2,3,4,a=1,b=2)



'''
Recursion

Recursive functions
'''

def factorial1(n):
    result = 1
    for i in range(n):
        result *= (i+1)
    return result    
    
    
   
def factorial2(n):
    if n == 1:
        return 1
    else:
        return n * factorial2(n-1)
        
     

'''
Categorizing and Tagging Words
'''

'''
The process of classifying words into their parts of speech and labeling them 
accordingly is known as part-of-speech tagging, POS-tagging, or simply tagging. 
Parts of speech are also known as word classes or lexical categories. 
The collection of tags used for a particular task is known as a tagset
'''        
        
'''
Using a Tagger
'''

import nltk
text = nltk.word_tokenize("And now for something completely different")        

nltk.pos_tag(text)
[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'),
('completely', 'RB'), ('different', 'JJ')]        

text = word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]


'''
text.similar() function

The text.similar() method takes a word w, finds all contexts w1w w2, then 
finds all words w' that appear in the same context, i.e. w1w'w2.
'''

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())  
text.similar('woman')
text.similar('bought')



'''
Tagged Corpora
'''
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token


sent = '''
... The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
... other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
... Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
... said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
... accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
... interest/NN of/IN both/ABX governments/NNS ''/'' ./.
... '''

[nltk.tag.str2tuple(t) for t in sent.split()]
 
 
'''
Reading Tagged Corpora
'''

nltk.corpus.brown.tagged_words()

nltk.corpus.brown.tagged_words(tagset='universal')


'''
Universal Part-of-Speech Tagset

Tag	Meaning	English Examples
ADJ	adjective	new, good, high, special, big, local
ADP	adposition	on, of, at, with, by, into, under
ADV	adverb	really, already, still, early, now
CONJ	conjunction	and, or, but, if, while, although
DET	determiner, article	the, a, some, most, every, no, which
NOUN	noun	year, home, costs, time, Africa
NUM	numeral	twenty-four, fourth, 1991, 14:24
PRT	particle	at, on, out, over per, that, up, with
PRON	pronoun	he, their, her, its, my, I, us
VERB	verb	is, say, told, given, playing, would
.	punctuation marks	. , ; !
X	other	ersatz, esprit, dunno, gr8, univeristy
'''


'''
Nouns
'''


from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()


'''
Word        After a determiner	                     Subject of the verb
woman	  the woman who I saw yesterday ...	      the woman sat down
Scotland	  the Scotland I remember as a child ...	 Scotland has five million people
book	       the book I bought yesterday ...	           this book recounts the colonization of Australia
intelligence the intelligence displayed by the child .  Mary's intelligence impressed her teachers
'''

word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
fdist = nltk.FreqDist(noun_preceders)
[tag for (tag, _) in fdist.most_common()]



'''
VERBS
'''

'''
Most common verbs
'''

 	
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB']
 
cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].most_common()
cfd1['cut'].most_common()


'''
reversing the key
'''


 	
wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
list(cfd2['VBN'])


'''
To clarify the distinction between VBD (past tense) and VBN (past participle), let's find words which can be both VBD and VBN, and see some surrounding text
'''

[w for w in cfd1.conditions() if 'VBD' in cfd1[w] and 'VBN' in cfd1[w]]
 
idx1 = wsj.index(('kicked', 'VBD'))
wsj[idx1-4:idx1+1]

idx2 = wsj.index(('kicked', 'VBN'))
wsj[idx2-4:idx2+1]


'''
Adjectives and Adverbs
'''

'''
Two other important word classes are adjectives and adverbs. Adjectives 
describe nouns, and can be used as modifiers (e.g. large in the large pizza), 
or in predicates (e.g. the pizza is large). English adjectives can have 
internal structure (e.g. fall+ing in the falling stocks). Adverbs modify 
verbs to specify the time, manner, place or direction of the event described 
by the verb (e.g. quickly in the stocks fell quickly). Adverbs may also 
modify adjectives (e.g. really in Mary's teacher was really nice).

English has several categories of closed class words in addition to 
prepositions, such as articles (also often called determiners) (e.g., the, a),
modals (e.g., should, may), and personal pronouns (e.g., she, they). Each 
dictionary and grammar classifies these words differently.

'''


'''
finding most common nouns
'''

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())
    
tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))    
for tag in sorted(tagdict):
    print(tag, tagdict[tag])
    
    
'''
Exploring Tagged Corpora
'''

brown_learned_text = nltk.corpus.brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))


brown_lrnd_tagged = nltk.corpus.brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
d = nltk.FreqDist(tags)
d.tabulate()        

'''
Next, let's look at some larger context, and find words involving particular 
sequences of tags and words (in this case "<Verb> to <Verb>"). In 
code-three-word-phrase we consider each three-word window in the sentence, 
and check if they meet our criterion. If the tags match, we print the 
corresponding words.
'''

from nltk.corpus import brown
def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence): 
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)

for tagged_sent in nltk.corpus.brown.tagged_sents():
    process(tagged_sent)
    
    
'''
Finally, let's look for words that are highly ambiguous as to their part of 
speech tag. Understanding why such words are tagged as they are in each 
context can help us clarify the distinctions between the tags.   
'''

brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag)
                                for (word, tag) in brown_news_tagged) 
    

for word in sorted(data.conditions()):
    if len(data[word]) > 3:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, ' '.join(tags))
        
'''        
Linguistic Objects as Mappings from Keys to Values

Linguistic Object    Maps From	    Maps To
Document             Index	         Word	List of pages (where word is found)
Thesaurus	           Word sense	    List of synonyms
Dictionary	      Headword	         Entry (part-of-speech, sense definitions, etymology)
Comparative Wordlist Gloss term	    Cognates (list of words, one per language)
Morph Analyzer	      Surface form	    Morphological analysis (list of component morphemes)        
'''


'''
Default Dictionaries
'''

'''
If we try to access a key that is not in a dictionary, we get an error. 
However, its often useful if a dictionary can automatically create an entry 
for this new key and give it a default value, such as zero or the empty list. 
For this reason, a special kind of dictionary called a defaultdict is 
available. In order to use it, we have to supply a parameter which can be 
used to create the default value, e.g.  int, float, str, list, dict, tuple.
'''

from collections import defaultdict
frequency = defaultdict(int)
frequency['colorless'] = 4
frequency['ideas']

pos = defaultdict(list)
pos['sleep'] = ['NOUN', 'VERB']
pos['ideas']


'''
The above examples specified the default value of a dictionary entry to be 
the default value of a particular data type. However, we can specify any 
default value we like, simply by providing the name of a function that can be 
called with no arguments to create the required value. Let's return to our 
part-of-speech example, and create a dictionary whose default value for any 
entry is 'N'. When we access a non-existent entry, it is automatically 
added to the dictionary 
'''

 	
pos = defaultdict(lambda: 'NOUN')
pos['colorless'] = 'ADJ'
pos['blog']

list(pos.items())

f = lambda: 'NOUN'
f()


def g():
    return 'NOUN'
g()



'''
We need to create a default dictionary that maps each word to its replacement. 
The most frequent n words will be mapped to themselves. Everything else will 
be mapped to UNK.
'''

alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v

alice2 = [mapping[v] for v in alice]
alice2[:100]

'''
Here's another instance of this pattern, where we index words according to their last two letters:
'''
 	
last_letters = defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

last_letters['ly']
last_letters['zy']


'''
The following example uses the same pattern to create an anagram dictionary. 
(You might experiment with the third line to get an idea of why this program 
works.)
'''


words = nltk.corpus.words.words('en') 	
anagrams = defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)

'''
different example to do the above task
'''   

anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)  


'''
Automatic Tagging
'''

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news') 


'''
Default Tagger : Gives the same tag to everyone
'''

raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)

'''
checking the performance against a tagged corpus
'''

default_tagger.evaluate(brown_tagged_sents)


'''
Regular Expression Tagger
'''

patterns = [
(r'.*ing$', 'VBG'),               # gerunds
(r'.*ed$', 'VBD'),                # simple past
(r'.*es$', 'VBZ'),                # 3rd singular present
(r'.*ould$', 'MD'),               # modals
(r'.*\'s$', 'NN$'),               # possessive nouns
(r'.*s$', 'NNS'),                 # plural nouns
(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
(r'.*', 'NN')                     # nouns (default)
]

regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)


'''
The Lookup Tagger

A lot of high-frequency words do not have the NN tag. Let's find the hundred 
most frequent words and store their most likely tag. We can then use this 
information as the model for a "lookup tagger" (an NLTK UnigramTagger):
'''    

 	
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

'''
Many words have been assigned a tag of None, because they were not among the 
100 most frequent words. In these cases we would like to assign the default 
tag of NN.
'''
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                   backoff=nltk.DefaultTagger('NN'))


'''
Let's put all this together and write a program to create and evaluate lookup 
taggers having a range of sizes
'''

def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
    
display()    


'''
N-Gram Tagging
'''
'''
Unigram Tagging

Unigram taggers are based on a simple statistical algorithm: for each token, 
assign the tag that is most likely for that particular token
'''

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'),
('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'),
(',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'),
('floor', 'NN'), ('so', 'QL'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'),
('direct', 'JJ'), ('.', '.')]
unigram_tagger.evaluate(brown_tagged_sents)
0.9349006503968017



'''
Separating the Training and Testing Data
'''

brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
size
4160
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)



'''
General N Gram Tagging
'''

'''
An n-gram tagger is a generalization of a unigram tagger whose context is the 
current word together with the part-of-speech tags of the n-1 preceding tokens, 
as shown in 5.1. The tag to be chosen, tn, is circled, and the context is 
shaded in grey. In the example of an n-gram tagger shown in 5.1, we have n=3; 
that is, we consider the tags of the two preceding words in addition to the 
current word. An n-gram tagger picks the tag that is most likely in the given 
context.
'''

bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])
unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)
bigram_tagger.evaluate(test_sents)


'''
Combining Taggers
'''

'''
One way to address the trade-off between accuracy and coverage is to use the 
more accurate algorithms when we can, but to fall back on algorithms with 
wider coverage when necessary. For example, we could combine the results of a 
bigram tagger, a unigram tagger, and a default tagger, as follows:
'''
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)
0.844911791089405

'''
Storing Taggers: store it as pickle
'''

#saving it as pickle 	
from pickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

#loading the model from a pickle
from pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()



'''
ambigutiy captured
'''
cfd = nltk.ConditionalFreqDist(
        ((x[1], y[1], z[0]), z[1])
        for sent in brown_tagged_sents
        for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()
0.049297702068029296


'''
Some morphosyntactic distinctions in the Brown tagset

Form	       Category	           Tag
go	       base	                VB
goes	       3rd singular present	 VBZ
gone	       past participle	      VBN
going	  gerund	                VBG
went	       simple past	           VBD
'''
