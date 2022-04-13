#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:57:58 2021

@author: nour
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 00:54:53 2021

@author: joe
"""


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import os
import math
import numpy as np
from prettytable import PrettyTable 
import operator


def Zero_appending1(ar1,ar2,ar3):
    ar1.append(0)
    ar2.append(0)
    ar3.append(0)
#open files and scan them
path = '/home/nour/Desktop/Documents' #Documents Folder Path
os.chdir(path)
my_Documents=[]
noOfDocs=0
#loop through files in the dir
for file in sorted(os.listdir())    :
    if file.endswith(".txt"): #if file is text file open it
        file_path=f"{path}/{file}"
        f= open(file_path, 'r')
        #read file and replace new line with empty so '\n' doesn't get copied
        #store read line in my_Documents
        my_Documents.append(f.read())
        noOfDocs+=1
# PASHE 1 #
stopW = set(stopwords.words("english"))
stopW.remove("in")
stopW.remove("where")
stopW.remove("to")


i=0
for x in my_Documents:
     my_Documents[i]=list(word_tokenize(x.lower()))  # Casefolding for all Words to match them with the Stop words
     my_Documents[i]=[word for word in my_Documents[i] if not word in stopW]
     i+=1   
 # PASHE 1 #
 
#PHASE 2#

dict1=defaultdict(lambda: defaultdict(list)) #Auxilarity Structure
docID=0
count=0
for docs in my_Documents:
    for words in my_Documents[docID]:
        dict1[words][docID].append(count)
        count+=1
    docID+=1
    count=0
# above is making the positional index

#term frequency and Tf_Wt
#make dict for term freq with noOfDocs keys

i=0
tfDoc={}
terms=[]
freq=[]
doc_tf_wt=[]
while i<noOfDocs:
    tfDoc[i]={}
    freq.append([])
    i+=1

docId=0
while docId<noOfDocs:
    for term in my_Documents[docId]:
        tfDoc[docId][term]=my_Documents[docId].count(term)
        terms.append(term)
    docId+=1
#print(tfDoc)

# PLotting Tf and Tf_Wt
columns=['Document %d' % (x+1) for x in range(noOfDocs)]
terms=list(sorted(set(terms)))
docId=0
while docId<noOfDocs:
    for term in terms:
       try:
           freq[docId].append(tfDoc[docId][term])
           doc_tf_wt.append(1+math.log(tfDoc[docId][term],10))
       except KeyError:
            freq[docId].append(0)
            doc_tf_wt.append(0)
    docId+=1
myTable=PrettyTable()
myTable22=PrettyTable()
myTable.add_column("Words",terms)
myTable22.add_column("Words",terms)
row=[]
row2=[]
col=0
j=0
for fr in freq:
    for index in fr:
       row.append(index)
       row2.append(round(doc_tf_wt[j],2))
       j+=1
    myTable.add_column(columns[col],row)
    myTable22.add_column(columns[col],row2)
    col+=1
    row=[]
    row2=[]
print("              Term Frequency        ")
print(myTable)
print("Tf Weighted")
print(myTable22)

#Plotting Tf


#document freq
#make dict that stores the df called dFreq
""""""
freqq=[0] *len(terms)
idf=[0]* len(terms)
dFreq ={}
docId=0
while docId<noOfDocs:
    for term in tfDoc[docId].keys():
        dFreq[term]={}
       
    docId+=1

docId=0
freq=0

for words in dict1.keys():
   freq=0
   docId=0
   while docId<noOfDocs:
      freq+=len(dict1[words][docId])
      dFreq[words]=freq
      try:
          j=terms.index(words)
          freqq[j]=dFreq[words]
      except ValueError:
             pass
      docId+=1

#DFreq Plotting
myTable2=PrettyTable()
myTable2.add_column("Words",terms)
myTable2.add_column("DF",freqq)
#DFreq Plotting

#idf
dictIdf ={}
docId=0
while docId<noOfDocs:
    for term in tfDoc[docId].keys():
        dictIdf[term]={}
    docId+=1

for words in dFreq.keys():
    if(dFreq[words]>0):
        dictIdf[words]= math.log(noOfDocs/dFreq[words],10)
    else:
        dictIdf[words]=0
idf_col=[]
for term in terms:
    idf_col.append(round(dictIdf[term],2))
myTable2.add_column("IDF",idf_col)
print("tf-idf")
print(myTable2)
#print(dictIdf)

#Tf-IDF
docId=0
myTable33=PrettyTable()
myTable33.add_column("Words",terms)
doc_wt=[]
while docId<noOfDocs:
    doc_wt.append([])
    for term in terms:
        if(tfDoc[docId].get(term)):
            doc_wt[docId].append(round((1+math.log(tfDoc[docId][term],10))*dictIdf[term],2))
        else:
            doc_wt[docId].append(0)
    myTable33.add_column(columns[docId],doc_wt[docId])
    docId+=1
print("tf*idf")
print(myTable33)

#Docs length and normalized tf-idf
myTable4=PrettyTable(["Document","Document Length"])
docId=0
doc_len=[]
while docId < noOfDocs:
    strr="Document"+str(docId+1)
    myTable4.add_row([strr,round(np.sqrt(np.sum(np.square(doc_wt[docId]))),2)])
    doc_len.append(np.sqrt(np.sum(np.square(doc_wt[docId]))))
    docId+=1
print("Documents Length")
print(myTable4)

#Normalized tf-idf
docId=0
myTable5=PrettyTable()
while docId < noOfDocs:
    myTable5.add_column(columns[docId],np.round(np.divide(doc_wt[docId],doc_len[docId]),2))
    docId+=1
print(myTable5)



# phrase query
query = input("Enter the query: ")    
queryTokenizedNoStop = word_tokenize(query)
queryTokenizedNoStop = [x.lower() for x in queryTokenizedNoStop ]
queryTokenizedNoStop = [word for word in queryTokenizedNoStop if not word in stopW]
# here to pre-processing the query
i=0 
docID=0
docs =[]
for term in queryTokenizedNoStop:
        docID=list(((dict1[term].keys())))
        docs.append(docID)

docs.sort(key=len) #here to sort el docs like frequency terms
#print(dict1)
#print(dict1[term].keys())
#print(docs)
intersectionDocs = list(set.intersection(*map(set,docs))) #here get the intersection documents
#print(intersectionDocs)

listOfIntersectedValue =[]
newList=[]
docIntersectionID = []
indexinindex =0
listCheckIntersection =[]
intersectionDocs = []
if len (queryTokenizedNoStop) !=0:
    intersectionDocs = list(set.intersection(*map(set,docs))) #here get the intersection documents
else:
    print("Query contains stop words only ")
    
if len(intersectionDocs) ==0:
    print("No matching documents")
else:
    for y in range(len(intersectionDocs)):
        for x in queryTokenizedNoStop:
            listOfIntersectedValue.append(dict1[x][intersectionDocs[y]])
        for z in range(len(listOfIntersectedValue)):
            for m in range (len(listOfIntersectedValue[z])):
                    listOfIntersectedValue[z][m] =  listOfIntersectedValue[z][m] - z
            listCheckIntersection=(list(set.intersection(*map(set,listOfIntersectedValue))))
            if len(listCheckIntersection) != 0:
                docIntersectionID.append(intersectionDocs[y])
                listCheckIntersection =[]           
        listOfIntersectedValue=[]

    docIntersectionNoDuplicate = []
    [docIntersectionNoDuplicate.append(x) for x in docIntersectionID if x not in docIntersectionNoDuplicate]
    
    #Tf-IDf
        
    doc_tf=[]
    doc_tf_wt=[]
    doc_wt=[]
    query_wt=[]
    query_tf=[]
    query_tf_wt=[]
    Terms=[]
    df=[]
    idf=[] 
    query_nize=[]
    doc_nize=[]
    docId=0
    Query_len=0
    Doc_len=0
    cos_sim={}
    idf=[]
    myTable3=PrettyTable()
    for docId in docIntersectionNoDuplicate :
            Terms=list(tfDoc[docId].keys())
            Terms=sorted(set(Terms+list(queryTokenizedNoStop)))
            for terms in Terms:
                # Query WT
                if(queryTokenizedNoStop.count(terms)>0):    
                        query_tf.append(queryTokenizedNoStop.count(terms))
                        query_tf_wt.append(1+math.log(queryTokenizedNoStop.count(terms),10))
                        query_wt.append(((1+math.log(queryTokenizedNoStop.count(terms),10)) *dictIdf[terms]))
                        
                else:
                        Zero_appending1(query_tf,query_tf_wt,query_wt)
                #Doc WT
                if(tfDoc[docId].get(terms)):
                    doc_tf.append(tfDoc[docId][terms])
                    doc_tf_wt.append(1+math.log(tfDoc[docId][terms],10))
                    doc_wt.append((1+math.log(tfDoc[docId][terms],10))*dictIdf[terms])
                    idf.append(dictIdf[terms])
                    df.append(dFreq[terms])
                else:
                   doc_tf.append(0)
                   doc_tf_wt.append(0)
                   doc_wt.append(0)
                   idf.append(0)
     
            Query_len=np.sqrt(np.sum(np.square(query_wt)))
            query_nize=np.divide(query_wt,Query_len)
            doc_len=np.sqrt(np.sum(np.square(doc_wt)))     
            doc_nize=np.divide(doc_wt,doc_len)
            myTable3.add_column("Terms",Terms)
            myTable3.add_column("Query Tf-r", query_tf)
            myTable3.add_column("Query Tf-wt", query_tf_wt)
            myTable3.add_column("Query Wt", np.around(query_wt,2))
            myTable3.add_column("Query n'lize",np.around(query_nize,2))
            myTable3.add_column("Df",df)
            myTable3.add_column("Idf",np.around(idf,2))
            myTable3.add_column("Doc Tf-r",doc_tf)
            myTable3.add_column("Doc Tf-wt",doc_tf_wt)
            myTable3.add_column("Doc Wt",np.around(doc_wt,2))
            myTable3.add_column("Doc n'lize",np.around(doc_nize,2))
            myTable3.add_column("Prod",np.around(np.multiply(doc_nize,query_nize),2))
            #Tf-Idf Table
            print("Document",(docId+1))    
            print(myTable3)
            cosine_similarity=np.sum(np.multiply(doc_nize,query_nize))
            cos_sim[docId+1]=cosine_similarity
            print("Cosine Similarity ",cosine_similarity)
            myTable3=PrettyTable()
            docId+=1
            myTable3=PrettyTable()
            doc_tf=[]
            doc_tf_wt=[]
            doc_wt=[]
            query_wt=[]
            query_tf=[]
            query_tf_wt=[]
            Terms=[]
            df=[]
            idf=[]
            query_nize=[]
            doc_nize=[]
    sorted_d = dict( sorted(cos_sim.items(), key=operator.itemgetter(1),reverse=True))
    
    print("Ranked documents relevant to query :",end=" ")
    for doc in sorted_d:
        print(doc,end=", ")

