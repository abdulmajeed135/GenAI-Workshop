# importing the Dataset

import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


# Sample text
sample_text = "Even my brother is not like to speak with me. They treat me like aids patent."

# Preprocess the sample text
sample_text = re.sub('[^a-zA-Z]', ' ', sample_text)
sample_text = sample_text.lower()
sample_text = sample_text.split()
sample_text = [ps.stem(word) for word in sample_text if not word in stopwords.words('english')]
sample_text = ' '.join(sample_text)

# Transform the sample text using CountVectorizer
sample_vector = cv.transform([sample_text]).toarray()

# Make predictions
prediction = spam_detect_model.predict(sample_vector)

# Interpret the prediction
if prediction[0] == 1:
    print("The sample text is predicted as spam.")
else:
    print("The sample text is predicted as not spam.")