from nltk.tokenize import word_tokenize
from typing import final
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import math
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
query_index={}
cosine_sim=[]
def ProcessQuery(query_tokens):
    print("Processing Query")
    connecing_words = []
    query_words = []
    filtered_query=[]
    for w in total_words:
        query_index[w]=0
    for w in query_tokens:
        if w not in stopList:
            filtered_query.append(w)

    for w in filtered_query:
            lemma_query = lemma.lemmatize(w)
            i = filtered_query.index(w)
            filtered_query[i] = lemma_query

    for w in filtered_query:
        query_index[w]=1

def calculate_cosine_sim(alpha):
    x = 0.0
    a = 0.0
    b = 0.0
    final = []
    for doc_num in range(1,449):
        for word in query_index:
            x += (query_index[word] * postings[word]['tf-idf'][doc_num]) 
            a += query_index[word]**2
            b += postings[word]['tf-idf'][doc_num]**2
        cosine_sim.append(x / (math.sqrt(a) * math.sqrt(b) ) )
        x = 0.0
        a = 0.0
        b = 0.0
    doc = []
    f=1
    for i in cosine_sim:
        if i > alpha:
            final.append(i)
            doc.append(f)
        f = f +1
    
    return final, doc
    


f = open("D:/UNIVERSITY/SEMESTER 6/IR/A1/Stopword-List.txt", "r")
stopList = f.read()
docList = 0
postings = {}
positional_index = {}
total_words=[]
# ps = PorterStemmer()
lemma = WordNetLemmatizer()
count = 0
for i in range(1, 449):
    f = open(f"D:/UNIVERSITY/SEMESTER 6/IR/A1/Abstracts/{i}.txt", "r")
    docList = f.read()
    docList = docList.replace("\n", " ")
    docList = docList.replace('-', " ")
    docList = docList.replace("/", " ")
    tokens = nltk.word_tokenize(docList)
    tokens = [tokens.lower() for tokens in tokens if tokens.isalnum()]
    filtered_sentence = [w for w in tokens if not w.lower() in stopList]
    filtered_sentence = []
    for w in tokens:
        if w not in stopList:
            filtered_sentence.append(w)
  
    for w in filtered_sentence:
        lemma_word=lemma.lemmatize(w)
        if lemma_word not in total_words:
            total_words.append(lemma_word)
        if lemma_word not in postings:
            postings[lemma_word]={
                        'tf' : [0]*449,  #list of termfreq for each doc
                        'df' : 0,
                        'idf':0,
                        'tf-idf':[0]*449  # list of tf-idf for each doc
                    }
            postings[lemma_word]['tf'][i] = 1
            postings[lemma_word]['df'] = 1
        else:
            if postings[lemma_word]['tf'][i] == 0 :  #found in new doc
                postings[lemma_word]['df'] = postings[lemma_word]['df'] + 1
                postings[lemma_word]['tf'][i] = 1
            else :
                postings[lemma_word]['tf'][i] = postings[lemma_word]['tf'][i] +1 

for doc_num in range(1,449):
        for word in postings:
            postings[word]['idf'] = math.log(449/(postings[word]['df']) , 10 )     #formula is log(N/df) not log(df/N)
            postings[word]['tf-idf'][doc_num] = postings[word]['tf'][doc_num] * postings[word]['idf']


# for key,value in postings.items():
#     print(key,":",value)

f = open("inverted_index.txt", "w")
f.write(str(postings))
f.close()

f=open("query_index.txt","w")
f.write(str(query_index))
f.close()
def takeInput(query):
    query_tokens=nltk.word_tokenize(query)
    query_tokens=[query_tokens.lower() for query_tokens in query_tokens if query_tokens.isalnum()]
    ProcessQuery(query_tokens)
    cosine_sim_list,doc_list=calculate_cosine_sim(0.001)
    print("doc",doc_list)
    print("final",cosine_sim_list)
    return cosine_sim_list,doc_list


    
