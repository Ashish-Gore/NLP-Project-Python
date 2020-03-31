import pandas as pd
import numpy as np
import re
import collections
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('white')
import nltk
#nltk.download('all')
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
#nltk.download('averaged_perceptron_tagger')

##################loading Data###########
data= pd.read_csv("data_file.csv")
##dataset information
data.head()
data.columns
data.shape
data.info()

#################################Exploratory Data Analysis######################################
## column wise data exploration
data_asin=data.groupby("asin").count()
data_asin
data_questionType=data.groupby('questionType').count()
data_questionType
data_answerType=data.groupby('answerType').count()
data_answerType
###finding Number of unique values
data.nunique()

##Checking Null values
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis');plt.title("Column wise Null values")

data.fillna("0", inplace=True) #replacing null values by '0' string
data.isnull().sum()

##considering Question and Answer Columns
data1=data.iloc[:,[4,6]]
data1.columns
question=data1.question
answer=data1.answer
question.head()
answer.head()

question.isnull().sum()
answer.isnull().sum()



#############################Basic feature extraction of text data
## number of characters
data1['que_char_count']=question.str.len()
data1['ans_char_count']=answer.str.len()

sns.set()
plt.hist(x='que_char_count', data=data1, bins=20)
plt.xlabel('Question Text Length')
plt.ylabel('Number of Questions')
plt.figsize=(10, 16)

bin_edges=[0, 100 ,200, 300, 400, 500, 600, 700, 800, 900, 1000]
plt.hist(x = 'ans_char_count', data=data1, bins=bin_edges)
plt.xlabel('Answer Text Length')
plt.ylabel('Number of Answer')
plt.figsize=(10, 16)

##number of words
data1['que_word_count']=question.apply(lambda x: len(str(x).split(" ")))
data1.head()
data1['ans_word_count']=answer.apply(lambda x: len(str(x).split(" ")))
data1.head()

sns.set()
plt.hist(x='que_word_count', data=data1, bins=20)
plt.xlabel('Word count')
plt.ylabel('Number of Questions')
plt.figsize=(10, 16)

sns.set()
bin_edges=[0,50,100,150,200,250,300]
plt.hist(x='ans_word_count', data=data1, bins=bin_edges)
plt.xlabel('Word Count')
plt.ylabel('Number of Answer')
plt.figsize=(10, 16)


### average word length
def avg_word(sentence):
    words=sentence.split(" ")
    return(sum(len(word) for word in words)/len(words))
    
data1['que_avg_word']=question.apply(lambda x: avg_word(x))
data1['ans_avg_word']=answer.apply(lambda x:avg_word(x))

## length of stopwords
from nltk.corpus import stopwords
with open("stop.txt", "r") as sw:
    stop_words=sw.read()

stop = stop_words.split("\n")
stop=stopwords.words("english")
data1['que_stopwords']=question.apply(lambda x: len([x for x in x.split() if x in stop]))
data1['ans_stopwords']=answer.apply(lambda x: len([x for x in x.split() if x in stop]))

### number of special characters
data1['que_special']=question.apply(lambda x:sum(x.count(s) for s in '#!*$&@%?+-^.,'))
data1['ans_special']=answer.apply(lambda x:sum(x.count(s) for s in '#!*$&@%?+-^.,' ))

## number of numerics
data1['que_numerics']=question.apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data1['ans_numerics']=answer.apply(lambda x: len([x for x in x.split() if x.isdigit()]))

## number of uppercase words
data1['que_uppercase']=question.apply(lambda x: len([x for x in x.split() if x.isupper()]))
data1['ans_uppercase']=answer.apply(lambda x: len([x for x in x.split() if x.isupper()]))




######################## Basic text preprocessing ########################## 

##lower case
data1['Question']=question.apply(lambda x:" ".join(x.lower() for x in x.split()))
data1['Answer']=answer.apply(lambda x:" ".join(x.lower() for x in x.split()))

## removing punctuations
data1['Question']=data1['Question'].str.replace('[^\w\s]','')
data1['Answer']=data1['Answer'].str.replace('[^\w\s]','')

##removal of digits
data1['Question']=data1['Question'].str.replace('\d+', '')
data1['Answer']=data1['Answer'].str.replace('\d+', '')

## removal of stopwords
with open("stop.txt", "r") as sw:
    stop_words=sw.read()

stop = stop_words.split("\n")

data1['Question']=data1['Question'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))
data1['Answer']=data1['Answer'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))


##removal of white spaces 
data1['Question']=data1['Question'].str.strip()
data1['Answer']=data1['Answer'].str.strip()

####Lemmatization
data1['Question'] = data1['Question'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
data1['Answer']=data1['Answer'].apply(lambda x: " ".join([lemmatizer.lemmatize(word)for word in x.split()]))

###top 10 most frequent repeated words in Question & Answer
freq1 = pd.Series(' '.join(data1['Question']).split()).value_counts()[:10]
freq1
freq2 = pd.Series(' '.join(data1['Answer']).split()).value_counts()[:10]
freq2


########### wordcloud generation ###############
####Question
Question_text = ' '.join(data1['Question'])
Q_wordcloud=WordCloud(
                    background_color='white',
                    width=1800,
                    height=1400
                   ).generate(Question_text)
fig = plt.figure(figsize = (10, 15))
plt.axis('off')
plt.imshow(Q_wordcloud)

###Answer
Answer_text = ' '.join(data1['Answer'])
A_wordcloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(str(Answer_text))
fig = plt.figure(figsize = (10, 15))
plt.axis('off')
plt.imshow(A_wordcloud )


#### parts of speech tagging
def pos_tagging(text):
    text_split=" ".join(text)
    text_splitted=text_split.split()
    text_splitted=[s for s in text_splitted if s]
    pos_list = nltk.pos_tag(text_splitted)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    print(pos_list)
    print("Noun_count:", noun_count)
    print("adjective_count:" ,adjective_count)
    print("verb_count:" ,verb_count )
   
    
pos_tagging(data1['Question'])
pos_tagging(data1['Answer'])




################# bigrams for Question and Answers 
for i in data1['Question'][0:10]:
    grams=TextBlob(i).ngrams(2)
    print(grams)
    
counts = collections.Counter()
for i in data1['Question']:
    words1 =i.split()
    counts.update(nltk.bigrams(words1))
    
common_bigrams = counts.most_common(10)
common_bigrams

##Answer
for i in data1['Answer'][0:10]:
    grams=TextBlob(i).ngrams(2)
    print(grams)
    
counts = collections.Counter()
for i in data1['Answer']:
    words1 =i.split()
    counts.update(nltk.bigrams(words1))
    
common_bigrams = counts.most_common(10)
common_bigrams

################## TF-IDF #########################

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(data1['Question'])
vect1=tfidf.fit_transform(data1['Answer'])
print(vect)
print(vect1)
vect

######### Bag of words##############
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(data1['Question'])
print(train_bow)

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(data1['Answer'])
print(train_bow)


######################################### sentiment analysis##############################
data1['Q_sentiment'] = data1['Question'].apply(lambda x: TextBlob(x).sentiment[0] )
data1[['Question','Q_sentiment']].head()
data1['A_sentiment']=data1['Answer'].apply(lambda x: TextBlob(x).sentiment[0])
data1[['Answer','A_sentiment']].head()


def sent_type(text):
    for i in text:
        if i>0:
            print('positive')
        elif i==0:
             print('natural')
        else:
             print('negative')
                 
           
sent_type(data1['Q_sentiment'])
sent_type(data1['A_sentiment'])

print('5 random Questions with the highest Positive sentiment polarity: \n')
most_positive = data1.loc[data1.Q_sentiment == 1, ['Question']].sample(5).values
for c in most_positive:
    print(c[0])
    
print('5 random Questions with the highest Negative sentiment polarity: \n')
most_neg= data1.loc[data1.Q_sentiment == -1, ['Question']].sample(5).values
for c in most_neg:
    print(c[0])

print('5 random Answers with the highest Positive sentiment polarity: \n')
most_pos= data1.loc[data1.A_sentiment == 1, ['Answer']].sample(5).values
for c in most_pos:
    print(c[0])
   
print('5 random Answer with the highest Negative sentiment polarity: \n')
most_neg= data1.loc[data1.Q_sentiment == -1, ['Answer']].sample(5).values
for c in most_neg:
    print(c[0])

  

sns.set()
plt.hist(x='Q_sentiment', data=data1, bins=20);
plt.xlabel('polarity of Questions');
plt.ylabel('count'); 
plt.figsize=(10, 16)



sns.set()
plt.hist(x='A_sentiment', data=data1, bins=20);
plt.xlabel('polarity of Answer');
plt.ylabel('count'); 
plt.figsize=(10, 16)



























