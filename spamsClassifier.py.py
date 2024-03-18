import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)


# In[4]:


import os


# In[6]:


os.chdir("C:\\Users\Morvi panchal\OneDrive\Desktop\ML_Proejct_codesoft")


# In[9]:


df = pd.read_csv("spam.csv",encoding='latin1')


# In[10]:


df.head()


# In[11]:


df2 = df[['v1','v2']]


# In[12]:


df2['processed_text'] = df2['v2'].apply(preprocess_text)


# In[13]:


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_features = tfidf_vectorizer.fit_transform(df2['processed_text']).toarray()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df2['v1'], test_size=0.2, random_state=42)


# In[15]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[16]:


with open('logreg_model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)


# In[17]:


with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)


# In[22]:




# In[ ]:





# In[25]:


def main():
    st.title("SMS Spam Classification")
    msg_text = st.text_area("Enter the SMS message you want to classify:", "Type here...")
    with open('logreg_model.pkl', 'rb') as model_file:
        lg = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vec = pickle.load(vectorizer_file)

    if st.button("Classify"):
        processed_message = preprocess_text(msg_text)
        tfidf_message = tfidf_vec.transform([processed_message]).toarray()
        prediction = lg.predict(tfidf_message)
        
        if prediction[0] == 'spam':
            st.write("This message is SPAM")
        else:
            st.write("This message is not SPAM")

if __name__ == "__main__":
    main()


# In[ ]:




