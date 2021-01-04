# Sentiment-Analysis-on-IMDb-reviews
Sentiment Analysis using Logistic regression performed on IMDb movie reviews.

# Required Libraries and Packages
* import numpy as np # linear algebra
* import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
* from textblob import TextBlob
* from wordcloud import WordCloud,STOPWORDS
* import matplotlib.pyplot as plt
* from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
* from nltk import word_tokenize
* import string
* from langdetect import detect_langs
* from nltk.corpus import words
* from nltk.tokenize import TweetTokenizer
* from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
* from nltk.stem import PorterStemmer
* from sklearn.preprocessing import StandardScaler
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LogisticRegression
* from sklearn.metrics import accuracy_score
* from nltk.corpus import stopwords
* from textblob import TextBlob

### To preview dataset click [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Reading data and converting 'sentiment' column to numerical type
```
  data=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
  data['sentiment_label']=[1 if x=='positive' else 0 for x in data['sentiment']]
  data = data.drop('sentiment', 1)
```
## Preparing word cloud of positive and negative reviews to get gist of most repeated words in the review column
### Positive reviews word cloud
```
#word cloud without stop words
#creating positive reviews string
positive_reviews=[]
for index, row in data.iterrows():
   if row['sentiment_label']==1:
      positive_reviews.append(row['review'])
des=""
des=des.join(positive_reviews)
my_cloud_positive_reviews = WordCloud(background_color='white').generate(des)
plt.imshow(my_cloud_positive_reviews, interpolation='bilinear') 
plt.axis("off")
plt.show()
```
* Output image click ![here](image.jpg)
