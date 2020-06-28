# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:41:41 2020

@author: Sayantan
"""
# Importing the libraries
import pandas as pd
import string
%matplotlib inline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import tokenize
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Loading the datasets
true_df = pd.read_csv('FakeAndRealNewsDataset_True.csv')
fake_df = pd.read_csv('FakeAndRealNewsDataset_Fake.csv')

# Creating 'check' on both dfs that will be the target feature.
true_df['check'] = 'TRUE'
fake_df['check'] = 'FAKE'

# We will combine both dfs.
df_news = pd.concat([true_df, fake_df])

# We will join title, text and subject to create the article feature
df_news['article'] = df_news['title']+""+df_news['text']+""+['subject']

# Creating the final Dataframe with article and check.
df = df_news[['article','check']]

# Converting to lower case
df['article'] = df['article'].apply(lambda x: x.lower())

# Removing punctuation
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str
df['article'] = df['article'].apply(punctuation_removal)

# Removing stopwords and converting to stem words
stop = stopwords.words('english')
df['article'].apply(lambda x: [item for item in x if item not in stop])
# ps = PorterStemmer()
# df = [ps.stem(word) for word in df]
          
# # Visualizing the data with Wordcloud
# all_words = ' '.join([text for text in df.article])

# wordcloud = WordCloud(width= 800, height= 500,
#                           max_font_size = 110,
#                           collocations = False).generate(all_words)


# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# # Function to generate wordcloud to True news.
# def wordcloud_true(text, column_text):
#     true_text = text.query("check == 'TRUE'")
#     all_words = ' '.join([text for text in true_text[column_text]])

#     wordcloud = WordCloud(width= 800, height= 500,
#                               max_font_size = 110,
#                               collocations = False).generate(all_words)
#     plt.figure(figsize=(10,7))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()
    
# # Function to generate wordcloud to Fake news.
# def wordcloud_fake(text, column_text):
#     fake_text = text.query("check == 'FAKE'")
#     all_words = ' '.join([text for text in fake_text[column_text]])

#     wordcloud = WordCloud(width= 800, height= 500,
#                               max_font_size = 110,
#                               collocations = False).generate(all_words)
#     plt.figure(figsize=(10,7))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()
    
# # Wordcloud of the true news.
# wordcloud_true(df, "article")

# # Wordcloud of the fake news.
# wordcloud_fake(df, "article")
# token_space = tokenize.WhitespaceTokenizer()
   

# def pareto(text, column_text, quantity):
#     all_words = ' '.join([text for text in text[column_text]])
#     token_phrase = token_space.tokenize(all_words)
#     frequency = nltk.FreqDist(token_phrase)
#     df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
#                                    "Frequency": list(frequency.values())})
#     df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
#     plt.figure(figsize=(12,8))
#     ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
#     ax.set(ylabel = "Count")
#     plt.show()
    
# #The 20 more frequent words.
# pareto(df, "article", 20)

#Lemmatization
'''from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in df["article"]]

df['article'] = df["article"].apply(lemmatize_text)'''

#Creating the bag of words
bow_article = CountVectorizer().fit(df['article'])
article_vect = bow_article.transform(df['article'])

#TF-IDF
tfidf_transformer = TfidfTransformer().fit(article_vect)
news_tfidf = tfidf_transformer.transform(article_vect)
print(news_tfidf.shape)

# Splitting the dataset into Training and Test sets
X = news_tfidf
y = df['check']
X_train, X_test, Y_train,Y_test= train_test_split(X, y, test_size=0.2)

#Naive Bayes model
fakenews_detector_NB = MultinomialNB().fit(X_train, Y_train)
predictions_NB = fakenews_detector_NB.predict(X_test)
print (classification_report(Y_test, predictions_NB))

# Support Vector Machine
fake_detector_svm = SGDClassifier().fit(X_train, Y_train)
predictions_svm = fake_detector_svm.predict(X_test)
print (classification_report(Y_test, predictions_svm))

# Logistic regression.
fake_detector_logreg = LogisticRegression().fit(X_train, Y_train)
predictions_logreg = fake_detector_logreg.predict(X_test)
print (classification_report(Y_test, predictions_logreg))

# Confusion Matrix
cm_NB = confusion_matrix(Y_test,predictions_NB)
accuracy_NB=(cm_NB[0][0]+cm_NB[1][1])/(cm_NB[0][0]+cm_NB[1][1]+cm_NB[0][1]+cm_NB[1][0])
cm_svm = confusion_matrix(Y_test,predictions_svm)
accuracy_svm=(cm_svm[0][0]+cm_svm[1][1])/(cm_svm[0][0]+cm_svm[1][1]+cm_svm[0][1]+cm_svm[1][0])
cm_logreg = confusion_matrix(Y_test,predictions_logreg)
accuracy_logreg=(cm_logreg[0][0]+cm_logreg[1][1])/(cm_logreg[0][0]+cm_logreg[1][1]+cm_logreg[0][1]+cm_logreg[1][0])


# cm = pd.DataFrame(cm , index = ['Fake','Not Fake'] , columns = ['Fake','Not Fake'])
# plt.figure(figsize = (10,10))
# sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Fake','Not Fake'] , yticklabels = ['Fake','Not Fake'])
# plt.xlabel("Actual")
# plt.ylabel("Predicted")

# cm = pd.DataFrame(cm , index = ['Fake','Not Fake'] , columns = ['Fake','Not Fake'])
# plt.figure(figsize = (10,10))
# sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Fake','Not Fake'] , yticklabels = ['Fake','Not Fake'])
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
