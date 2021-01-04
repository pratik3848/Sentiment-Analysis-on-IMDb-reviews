# Sentiment-Analysis-on-IMDb-reviews
Sentiment Analysis using Logistic regression performed on IMDb movie reviews.

# Required Libraries and Packages
```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
import string
from langdetect import detect_langs
from nltk.corpus import words
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from textblob import TextBlob

```
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
* Output image ![](https://user-images.githubusercontent.com/41427089/103519308-53eefa00-4e9b-11eb-8619-1bed5376ebdd.png)
```
#creating word cloud with stop words as per pervious output
my_stop_words = STOPWORDS.update(['films', 'film','br','movie','movies'])
my_cloud_positive_reviews = WordCloud(background_color='white', stopwords=my_stop_words).generate(des)
plt.imshow(my_cloud_positive_reviews, interpolation='bilinear') 
plt.axis("off")
plt.show()
```
* Output image ![](https://user-images.githubusercontent.com/41427089/103520754-aaf5ce80-4e9d-11eb-99f2-fc3e24bc321d.png)
### Negative reviews word cloud

```
#creating negative reviews string
negative_reviews=[]
for index, row in data.iterrows():
   if row['sentiment_label']==0:
    negative_reviews.append(row['review'])
neg=""
neg=neg.join(negative_reviews)
#word cloud without stop words
my_cloud_negative_reviews = WordCloud(background_color='white').generate(neg)
plt.imshow(my_cloud_negative_reviews, interpolation='bilinear') 
plt.axis("off")
plt.show()
```
* Output image ![](https://user-images.githubusercontent.com/41427089/103520766-ae895580-4e9d-11eb-8f79-574655c58f29.png)
```
#creating word cloud with stop words as per pervious output
my_negative_stop_words = STOPWORDS.update(['character','one','story','see'])
my_cloud_negative_reviews = WordCloud(background_color='white', stopwords=my_negative_stop_words).generate(neg)
plt.imshow(my_cloud_negative_reviews, interpolation='bilinear') 
plt.axis("off")
plt.show()
```
* Output image ![](https://user-images.githubusercontent.com/41427089/103520769-b0ebaf80-4e9d-11eb-9479-79a00323c8d8.png)
## Detecting review languages and filtering out any language other than English
```
#detecting review language
languages = []
# Loop over the sentences in the list and detect their language
for index,row in data.iterrows():
    languages.append(detect_langs(row['review']))
languages = [str(lang).split(':')[0][1:] for lang in languages]
data['languages']=languages
#keeping only english language reviews
data=data[data['languages']!='nl']
data=data[data['languages']!='id']
```
## Cleaning dataset
### Filtering out redundant words from the review column, keeping only words that are in English
```
#data cleaning- filtering out number and other redundant values
#removing all numbers as well as other redundant characters from reviews column
tknzr = TweetTokenizer()
tokenize_review = [tknzr.tokenize(item) for item in data.review]
cleaned_tokenized_reviews = [[word for word in item if word.isalnum()] for item in tokenize_review]
cleaned_tokenized_reviews = [[word for word in item if word.isalpha()] for item in tokenize_review]
set_words = set(words.words())
cleaned_tokenized_reviews = [[word for word in item if word in set_words] for item in cleaned_tokenized_reviews]
```
## Removing redundant words
```
stop_words = set(stopwords.words("english"))
stopwords_added=stop_words.union(['films', 'film','br','movie','movies','character','one','story','see'])
stopwords_removed_tokenize_reviews=[[word for word in item if word not in stopwords_added]for item in cleaned_tokenized_reviews]
str1=" "
converted_tokenized_reviews=[str1.join(item) for item in stopwords_removed_tokenize_reviews]
data['review']=converted_tokenized_reviews
```
## Adding extra features to dataset for better analysis
```
token_size=[len(item) for item in stopwords_removed_tokenize_reviews]
data['token_size']= token_size
```
## Building bag of words/vectorizer
```
# Build the vectorizer
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=200, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(data.review)
# Create sparse matrix from the vectorizer
X = vect.transform(data.review)
# Create a DataFrame
reviews_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
#adding sentiment and token_size column to new data frame
f_column = data[["token_size","sentiment_label"]]
reviews_transformed = pd.concat([reviews_transformed,f_column], axis = 1)
reviews_transformed.dropna(inplace=True)
```
## Scaling the columns and splitting data into train and test set
```
X = reviews_transformed.iloc[:, :-1].values
X.shape
y = reviews_transformed['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## Creating a logistic regression 
```
# Train a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)
# Predict the labels
y_predicted = log_reg.predict(X_test)
# Print accuracy score
print('Accuracy on the test set: ', accuracy_score(y_test,y_predicted) * 100)
```
* Output image ![](https://user-images.githubusercontent.com/41427089/103533073-793b3280-4eb2-11eb-87db-0f99f19af208.png)
