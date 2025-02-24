#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier  # Fixed import
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re 


# In[5]:


data = pd.read_csv(r"C:/Users/Sathw/Downloads/amazon_alexa.tsv", sep='\t') 
print(f"Dataset shape: {data.shape}")


# In[6]:


data.head(9)


# In[7]:


print(f" Feature names: {data.columns.values}")


# In[8]:


data.isnull().sum()


# In[9]:


data[data['verified_reviews'].isna()==True]


# In[11]:


data=data.dropna(inplace=True)


# In[15]:


# Verify if the file exists before reading
try:
    data = pd.read_csv(r"C:/Users/Sathw/Downloads/amazon_alexa.tsv", sep='\t')  # Use tab separator for TSV files
    print(f"Dataset shape before dropping null values: {data.shape}")

    # Drop null values and ensure it updates correctly
    data = data.dropna()  # Don't use inplace=True, assign the result back

    print(f"Dataset shape after dropping null values: {data.shape}")

except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file could not be parsed correctly. Check the format.")


# In[17]:


data['length'] = data['verified_reviews'].apply(len)


# In[19]:


data.head()


# In[22]:


# Print the review at index 10
print(f"'verified_reviews' column value: {data.iloc[10]['verified_reviews']}")

# Print the length of that review
print(f"Length of review: {len(data.iloc[10]['verified_reviews'])}")

# Print the corresponding 'length' column value
print(f"'length' column value: {data.iloc[10]['length']}")


# In[23]:


data.dtypes


# In[25]:


print(f"Rating value count:\n{data['rating'].value_counts()}")


# In[26]:


print(f"Rating value count-percentage distribution: \n{round(data['rating'].value_counts()/data.shape[0]*100,2)}")


# In[27]:


data['rating'].value_counts().plot.bar(color='red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# In[29]:


import matplotlib.pyplot as plt
from io import BytesIO

# Create figure
fig = plt.figure(figsize=(7,7))

# Define colors
colors = ['red', 'green', 'blue', 'orange', 'yellow']

# Define wedge properties
wp = {'linewidth': 1, 'edgecolor': 'black'}

# Get rating distribution as percentages
tags = data['rating'].value_counts() / data.shape[0]

# Define explode values for the pie chart
explode = (0.1, 0.1, 0.1, 0.1, 0.1)

# Plot pie chart
tags.plot(
    kind='pie', 
    autopct="%1.1f%%", 
    shadow=True, 
    colors=colors, 
    startangle=90, 
    wedgeprops=wp, 
    explode=explode
)

# Save the figure to a BytesIO object
graph = BytesIO()
fig.savefig(graph, format="png")

# Show the plot
plt.show()


# In[30]:


review_0 = data[data['feedback']==0].iloc[1]['verified_reviews']
print(review_0)


# In[32]:


review_1 = data[data['feedback']==1].iloc[1]['verified_reviews']
print(review_1)


# In[33]:


data['feedback'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()


# In[34]:


print(f"Variation value count:\n{data['variation'].value_counts()}")


# In[35]:


data['variation'].value_counts().plot.bar(color = 'orange')
plt.title('variation distribution')
plt.xlabel('variation')
plt.ylabel('count')
plt.show()


# In[37]:


import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(11,6))

# Group by 'variation', calculate mean rating, sort, and plot a bar chart
data.groupby('variation')['rating'].mean().sort_values().plot.bar(color='brown')

# Add labels and title
plt.title("Mean Rating According to Variation")
plt.xlabel("Variation")
plt.ylabel("Mean Rating")

# Show plot
plt.show()


# In[38]:


cv= CountVectorizer(stop_words='english')
words = cv.fit_transform(data.verified_reviews)


# In[42]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all reviews into a single string
reviews = " ".join([review for review in data['verified_reviews']])

# Create the WordCloud object
wc = WordCloud(background_color='white', max_words=50)

# Generate the word cloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))

# Add title and remove axes
plt.title(' Reviews', fontsize=15)
plt.axis('off')

# Show the plot
plt.show()


# In[44]:


# Ensure 'verified_reviews' column has no NaN values
data = data.dropna(subset=['verified_reviews'])

# Combine all reviews for each feedback category
neg_reviews = " ".join(data[data['feedback'] == 0]['verified_reviews'].str.lower())
pos_reviews = " ".join(data[data['feedback'] == 1]['verified_reviews'].str.lower())

# Convert to sets for faster lookup
neg_set = set(neg_reviews.split())
pos_set = set(pos_reviews.split())

# Finding words unique to each feedback category
unique_negative = " ".join(neg_set - pos_set)
unique_positive = " ".join(pos_set - neg_set)

# Print the first few words from each unique set for verification
print(f"Unique Negative Words: {unique_negative[:200]}...")
print(f"Unique Positive Words: {unique_positive[:200]}...")


# In[45]:


wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[46]:


wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[47]:


corpus = []
stemmer = PorterStemmer()
for i in range(0, data.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)


# In[48]:


cv = CountVectorizer(max_features = 2500)

#Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values


# In[50]:


import pickle
import os

# Ensure the directory exists
os.makedirs("Models", exist_ok=True)

# Save the CountVectorizer object
with open("Models/countVectorizer.pkl", "wb") as file:
    pickle.dump(cv, file)

print("CountVectorizer saved successfully!")


# In[51]:


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")


# In[53]:


print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")


# In[54]:


scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)


# In[55]:


#Saving the scaler model
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))


# In[56]:


#Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)


# In[57]:


#Accuracy of the model on training and testing data
 
print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))


# In[58]:


#Predicting on the test set
y_preds = model_rf.predict(X_test_scl)


# In[59]:


cm = confusion_matrix(y_test, y_preds)


# In[60]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
plt.show()


# In[61]:


accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())


# In[62]:


params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}


# In[63]:


cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())


# In[64]:


#Getting the best parameters from the grid search


print("Best Parameter Combination : {}".format(grid_search.best_params_))


# In[65]:


print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
print("Accuracy score for test set :", accuracy_score(y_test, y_preds))


# In[66]:


model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


# In[67]:


#Accuracy of the model on training and testing data
 
print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))


# In[68]:


y_preds = model_xgb.predict(X_test)


# In[69]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[70]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()


# In[ ]:




