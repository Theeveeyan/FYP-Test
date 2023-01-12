#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


# In[2]:


st.set_page_config(
    page_title="Sentiment Analysis with Term Weighting Techniques",
    page_icon="✅",
    layout="wide",
)


# In[3]:


st.title("Sentiment Analysis with Term Weighting Techniques")
option = st.selectbox(
    'Which dataset would you like to choose?',
    ('Amazon', 'IMDb', 'Yelp'))

st.write('You selected:', option)
st.set_option('deprecation.showPyplotGlobalUse', False)


# In[4]:


if option == 'Amazon':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/amazon_cells_labelled.txt"
elif option == 'IMDb':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/imdb_labelled.txt"
elif option == 'Yelp':
    filepath="https://raw.githubusercontent.com/Theeveeyan/FYP-Test/main/yelp_labelled.txt"
amazon_df=pd.read_csv(filepath,
                        delimiter='\t',
                        header=None, 
                        names=['review', 'sentiment'])


# Data Preprocessing

# In[5]:


amazon_df['review'] = amazon_df['review'].str.lower()


# In[6]:


amazon_df['review'] = amazon_df['review'].str.replace(r'[^\w\s]+', '')


# In[7]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[8]:


stop_words = stopwords.words('english')


# In[9]:


amazon_df['review'] = amazon_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[10]:


nltk.download('punkt');


# In[11]:


def tokenize(column):

    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]


# In[12]:


amazon_df['review'] = amazon_df.apply(lambda x: tokenize(x['review']), axis=1)


# In[13]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


# In[14]:


amazon_df['review'] = amazon_df['review'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[15]:


#Copy df for TF-IDF
amazon_copy = amazon_df.copy()


# In[16]:


st.dataframe(amazon_df)


# Data Modelling

# In[17]:


#Split dataset
train, test = train_test_split(amazon_copy, test_size=0.3, random_state=1)
X_train = train['review'].values
X_test = test['review'].values
y_train = train['sentiment']
y_test = test['sentiment']


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True, lowercase = False)
train_vectors = vectorizer.fit_transform([' '.join(review) for review in train['review']])
test_vectors = vectorizer.transform([' '.join(review) for review in test['review']])


# In[19]:


# Create the SVM model
svm = SVC(kernel='linear', probability=True)

# Train the SVM model on the selected features
svm.fit(train_vectors, y_train)


# In[20]:


#Copy df for IG
amazon_ig = amazon_df.copy()
train1, test1 = train_test_split(amazon_ig, test_size=0.3, random_state=1)
X_train1 = train1['review'].values
X_test1 = test1['review'].values
y_train1 = train1['sentiment']
y_test1 = test1['sentiment']


# In[21]:


from sklearn.feature_selection import mutual_info_classif

# Create feature vectors
vectorizer1 = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True, lowercase = False)
train_vectors1 = vectorizer1.fit_transform([' '.join(review) for review in train1['review']])
test_vectors1 = vectorizer1.transform([' '.join(review) for review in test1['review']])

# Select top k features using mutual information
k = 1000
mutual_info = mutual_info_classif(train_vectors1, train1['sentiment'], discrete_features=True)
top_k_idx = mutual_info.argsort()[-k:][::-1]

# Create new feature vectors with only top k features
train_vectors_new = train_vectors1[:, top_k_idx]
test_vectors_new = test_vectors1[:, top_k_idx]


# In[22]:


from sklearn.svm import SVC

# Build an SVM model
svm_model = SVC(kernel='linear', probability=True)

# Train the model
svm_model.fit(train_vectors_new, train1['sentiment'])


# Evaluation of both Models

# In[23]:


from sklearn.metrics import classification_report

# Make predictions on the test data
y_pred = svm.predict(test_vectors)

# Print the classification report
print("Model evaluation after using TF-IDF")
print(classification_report(y_test, y_pred))

# Make predictions on the test data
y_pred1 = svm.predict(test_vectors_new)

# Print the classification report
print("Model evaluation after using Information Gain")
print(classification_report(test1['sentiment'], y_pred1))


# In[24]:


accuracy1 = accuracy_score(y_test, y_pred)
accuracy2 = accuracy_score(test1['sentiment'], y_pred1)

import matplotlib.pyplot as plt

model_names = ['Model 1 TF-IDF', 'Model 2 Information Gain']
accuracy = [accuracy1, accuracy2]

fig2, ax = plt.subplots()
ax.bar(model_names, accuracy)
#ax.ylabel('Accuracy')

#st.pyplot(fig2)


# In[25]:


precision1 = precision_score(y_test, y_pred)
precision2 = precision_score(test1['sentiment'], y_pred1)

model_names = ['Model 1 TF-IDF', 'Model 2 Information Gain']
precision = [precision1, precision2]

fig3, ax = plt.subplots()
ax.bar(model_names, precision)
#ax.ylabel('Precision')

#st.pyplot(fig3)


# In[26]:


recall1 = recall_score(y_test, y_pred)
recall2 = recall_score(test1['sentiment'], y_pred1)

model_names = ['Model 1 TF-IDF', 'Model 2 Information Gain']
recall = [recall1, recall2]

fig4, ax = plt.subplots()
ax.bar(model_names, recall)
#ax.ylabel('Recall')

#st.pyplot(fig4)


# In[27]:


f11 = f1_score(y_test, y_pred)
f12 = f1_score(test1['sentiment'], y_pred1)

model_names = ['Model 1 TF-IDF', 'Model 2 Information Gain']
f1 = [f11, f12]

fig5, ax = plt.subplots()
ax.bar(model_names, f1)
#ax.ylabel('F1-Score')

#st.pyplot(fig5)


# In[28]:


fig_col1, fig_col2, fig_col3, fig_col4 = st.columns(4)
with fig_col1:
    st.markdown("### Accuracy")
    st.pyplot(fig2)
    
with fig_col2:
    st.markdown("### Precision")
    st.pyplot(fig3)
    
with fig_col3:
    st.markdown("### Recall")
    st.pyplot(fig4)
    
with fig_col4:
    st.markdown("### F1-Score")
    st.pyplot(fig5)


# In[29]:


# predict the probabilities of the positive class for each model
probs1 = svm.predict_proba(test_vectors)[:, 1]
probs2 = svm.predict_proba(test_vectors_new)[:, 1]

# calculate the AUC-ROC scores for each model
auc1 = roc_auc_score(y_test, probs1)
auc2 = roc_auc_score(y_test1, probs2)

# calculate the ROC curves for each model
fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1)
fpr2, tpr2, thresholds2 = roc_curve(y_test1, probs2)

# plot the ROC curves
fig = plt.figure()
plt.plot(fpr1, tpr1, label="Model 1 TF-IDF (AUC = %0.2f)" % auc1)
plt.plot(fpr2, tpr2, label="Model 2 Information Gain (AUC = %0.2f)" % auc2)

# add labels and title
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Models 1 and 2")
plt.legend(loc="lower right")

st.pyplot(fig)


# In[ ]:




