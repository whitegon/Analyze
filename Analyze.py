#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('Cp1252')

from collections import Counter
import pandas as pd

#read file
redditDF = pd.read_json('AllAboutTechhasReply_file.json',typ='frame', lines=True)

Post = 0
Reply = 0

#count Post and Reply
for data in redditDF['type'] :
    if( str( data ) == 'Post' ):
        Post = Post + 1
    else:
        Reply = Reply + 1

#count textcount
redditDF['textcount'] = [len( term ) for term in redditDF['text'] ]
redditDF['total'] = 1

#according to the author
dAndP = redditDF.groupby( 'author').aggregate( sum )

count_author = Counter()

#count author
for index in range( len(redditDF) ):
    if( str( redditDF['author'][index] ) != 'None' ) :
        # print (str( redditDF['author'][index] ))
        author_only = [str(redditDF['author'][index])]
        count_author.update(author_only)

print ( count_author.most_common(5) )

# -----------------------------------------------------------------

#補充遺漏值，不含內文的以標題代替
#fill no text  post with its title
for i in range( len(redditDF) ):
    if( len( redditDF['text'][i] ) == 0 ) :
        redditDF.set_value( i, ['text'], redditDF['title'][i]  )
        if( redditDF['text'][i] == "[removed]" or redditDF['text'][i] == "[deleted]" ):
            redditDF = redditDF.drop( i )

# print( redditDF['text'] )
# -----------------------------------------------------------------

#Tf-IDF
import nltk
import math
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

#Normalize by lemmatization(詞型還原)
# nltk.download() # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# print redditDF['text'][2]
# print ( LemNormalize(redditDF['text'][2]) )
# exit(0)

#停止詞，但TFIDF有內建的
# punctuation = list(string.punctuation)
# stop = stopwords.words('english') + punctuation + ['rt', 'via']

#TF-IDF向量化
vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') #TFIDF矩陣，並處理詞
tfidf = vectorizer.fit_transform(redditDF['text']) #特征提取
words = vectorizer.get_feature_names() #獲取詞語模型中的所有詞語(get the all words)


#印出每篇文章的tf-idf詞語權重，第一個for跑過所有文章，第二個for跑過某篇文章下的詞語權重(>0.7)
#print every text's tf-idf weight
for i in range(len(redditDF['text'])):
    print( redditDF['text'][i])
    for j in range(len(words)):
        if tfidf[i,j] > 0.7:
              print( words[j], tfidf[i,j])
# -----------------------------------------------------------------

#透過餘弦距離來算相似度
#(source code reference:https://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel #計算X和Y之間的線性內核

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten() #flatten()，使LIST攤平
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index] #argsort()，返回數組值從小到大的索引值
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

print( redditDF['text'][0]) #target text

for index, score in find_similar(tfidf, 0): #top_5 similar text
       print score, redditDF['text'][index], index


#情緒分析
from textblob import TextBlob
def SentimentAnalysis( text ) :
    blob = TextBlob(text)
    print(text)
    print blob.sentiment

SentimentAnalysis(redditDF['text'][3672])
#Sentiment(polarity=-0.4, subjectivity=0.6) 情感極性0.4， 主觀性0.6

# -----------------------------------------------------------------
#網路分析
import networkx as nx
G = nx.Graph()

for index  in range( len(redditDF) ):
    if( redditDF['type'][index] != 'Post' ):
        if( redditDF['author'][index] != 'None' and ( redditDF['ReplyToWho'][index] == 'izumi3682' or redditDF['ReplyToWho'][index] == 'yourSAS')   ):
            G.add_edge(redditDF['author'][index], redditDF['ReplyToWho'][index])

plt = nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =9,node_size =10)
plt.show()
#------------------------------------------------------------------------------------------------------------------------


import gc
del redditDF
gc.collect()

