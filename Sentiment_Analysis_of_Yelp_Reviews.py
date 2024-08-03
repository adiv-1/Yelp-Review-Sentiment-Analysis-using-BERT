#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install and Import Dependencies
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import math


# In[2]:


# Instantiating the model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[18]:


# Encoding and Calculating Sentiment
tokens = tokenizer.encode('Not sure if I would go there again, but the food was ok.', return_tensors = 'pt')
result = model(tokens)
result


# Highest value from Tensor results is going to represent what position represents the actual sentiment

# In[19]:


# Tensor results
result.logits


# In[21]:


# below is the tensor result from 0 to 4 (1 through 5) for the sentiment analysis
int(torch.argmax(result.logits))+1


# In[22]:


# Collecting reviews from Yelp
r = requests.get('https://www.yelp.com/biz/dishoom-london')
soup = BeautifulSoup(r.text,'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = {result.text for result in results}


# In[23]:


df = pd.DataFrame(reviews, columns=['review'])
df.head()


# In[13]:


df['review'].iloc[0]


# In[14]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors = 'pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[15]:


#Testing the function
sentiment_score(df['review'].iloc[1])


# In[16]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
df


# In[17]:


df['sentiment'].mean()

