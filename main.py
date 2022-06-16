import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

Train_Data = pd.read_csv('train.csv',sep='\t')
Test_Data = pd.read_csv('test.csv')
Train_Data.shape
(200000,2)
Train_Data[:5]

#Tfidf
tr_idf = TfidfVectorizer(max_features=2000).fit(Train_Data['text'].values)
train_tfidf = tr_idf.transform(Train_Data['text'].values)
train_tfidf.shape
(200000,2000)
test_tfidf = tr_idf.transform(Test_Data['text'].values)

clf = RidgeClassifier()
clf.fit(train_tfidf,Train_Data['label'].values)
RidgeClassifier()

val_pred = clf.predict(train_tfidf[10000:])
print(f1_score(Train_Data['label'].values[10000:],val_pred,average='macro'))

df = pd.DataFrame()
df['label'] = clf.predict(test_tfidf)
df.to_csv('cyf.csv',index=None)