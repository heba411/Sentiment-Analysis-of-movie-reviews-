import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


neg_path = "D:/myProjects/NLP_Project/Review_polarity_T005/review_polarity/txt_sentoken/neg"
pos_path = "D:/myProjects/NLP_Project/Review_polarity_T005/review_polarity/txt_sentoken/pos"

neg_reviews = []
for filename in os.listdir(neg_path):
    filepath = os.path.join(neg_path, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            content_neg = file.read()
            neg_reviews.append(content_neg)

neg_df = pd.DataFrame(neg_reviews, columns=['text'])
neg_df['sentiment'] = 'negative'

pos_reviews = []
for filename in os.listdir(pos_path):
    filepath = os.path.join(pos_path, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            content_pos = file.read()
            pos_reviews.append(content_pos)

pos_df = pd.DataFrame(pos_reviews, columns=['text'])
pos_df['sentiment'] = 'positive'

# Combine the positive and negative reviews into a single dataset
df = pd.concat([pos_df, neg_df], ignore_index=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# remove stopwords from dataset as a whole
stopword = set(stopwords.words('english')) - {'not'}


def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_list = [word for word in tokens if word.casefold() not in stopword]
    no_punct = [''.join(char for char in word if char not in string.punctuation) for word in filtered_list]
    no_punct = [word for word in no_punct if word]

    lemma_neg = WordNetLemmatizer()
    tags = pos_tag(no_punct)

    lemmatized_words = []
    for word, tag in tags:
        if tag.startswith('N'):
            lemmatized_word = lemma_neg.lemmatize(word, pos='n')  # noun
        elif tag.startswith('V'):
            lemmatized_word = lemma_neg.lemmatize(word, pos='v')  # verb
        elif tag.startswith('J'):
            lemmatized_word = lemma_neg.lemmatize(word, pos='a')  # adjective
        elif tag.startswith('R'):
            lemmatized_word = lemma_neg.lemmatize(word, pos='r')  # adverb
        else:
            lemmatized_word = lemma_neg.lemmatize(word)  # default to noun
        lemmatized_words.append(lemmatized_word)

    return ' '.join(lemmatized_words)

# Apply preprocessing to the text data
X_train_preprocessed = X_train.apply(preprocess_text)
X_test_preprocessed = X_test.apply(preprocess_text)

# Create a tf-idf vectorizer
tfidf = TfidfVectorizer()

# Fit the vectorizer on the preprocessed training data
tfidf.fit(X_train_preprocessed)

# Transform the preprocessed training and testing data into tf-idf vectors
X_train_tfidf = tfidf.transform(X_train_preprocessed)
X_test_tfidf = tfidf.transform(X_test_preprocessed)

#feature selection
selector = SelectKBest(chi2, k=1000)
X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf = selector.transform(X_test_tfidf)

############################################# Classifier ############################################################

# Create a logistic regression classifier
classifier = LogisticRegression(C=100)

# Train the classifier on the training data
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
predictions_logistic = classifier.predict(X_test_tfidf)

# Evaluate the accuracy of the classifier
accuracy_logistic = (predictions_logistic == y_test).mean()
print("Logistic Regression Accuracy:", accuracy_logistic)

#apply svm classifier
LinearSVM = SVC(kernel='linear', C=10)
LinearSVM.fit(X_train_tfidf, y_train)
y_predicted= LinearSVM.predict(X_test_tfidf)
accuracy_linear= accuracy_score(y_test, y_predicted)
print("SVM Accuracy with linear kernal function: ",accuracy_linear )


#svm with rbf
svm = SVC(kernel='rbf', gamma=1)
svm.fit(X_train_tfidf, y_train)

# Predict the class labels of the test data
y_pred = svm.predict(X_test_tfidf)

# Calculate the accuracy of the predictions
accuracy_rbf = accuracy_score(y_test, y_pred)
print('Accuracy with rbf',accuracy_rbf)

svm3= SVC(kernel='poly', degree=2)
svm3.fit(X_train_tfidf,y_train)

# Predict the class labels of the test data
y_pred_poly= svm3.predict(X_test_tfidf)

# Calculate the accuracy of the predictions
accuracy_poly= accuracy_score(y_test,y_pred_poly)
print('Accuracy with poly',accuracy_poly)

#apply KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_tfidf,y_train)

# Predict the class labels of the test data
y_pred_knn = knn.predict(X_test_tfidf)
knnAcc= accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy: ", knnAcc)

#random forest classifier
# Create a Random Forest classifier
classifier2 = RandomForestClassifier(min_samples_split=2)

# Train the classifier on the training data
classifier2.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
predictions2 = classifier2.predict(X_test_tfidf)

# Evaluate the accuracy of the classifier
accuracy2 = (predictions2 == y_test).mean()
print("Accuracy with Random Forest:", accuracy2)


# Create a confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Create a heatmap of the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()






# Create a classification report
report = classification_report(y_test, y_predicted)
print("Classification Report:",report)

#model accuracy comparison
# Create a list of model names
model_names = ['Logistic Regression','SVM Linear', 'SVM rbf', 'SVM poly', 'KNN', "Random forest"]

# Create a list of model accuracies
model_accs = [accuracy_logistic, accuracy_linear, accuracy_rbf, accuracy_poly, knnAcc, accuracy2]

# Plot the model accuracies
plt.bar(model_names, model_accs)

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

# Show plot
plt.show()









