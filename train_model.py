import pandas as pd # for data handling
import tensorflow as tf #neural network
import joblib # saving the model

from sklearn.model_selection import train_test_split #evaluation
from sklearn.preprocessing import LabelEncoder # convert to labels
from sklearn.feature_extraction.text import TfidfVectorizer #NLP feature extraction

# load dataset
df = pd.read_csv("D:/mental-health-cnn-project/data/final_dataset_small.csv") # dataset laoding

# create text column
df["text"] = df.astype(str).apply(" ".join, axis=1) # converting all colums to text string  joins thems to one long string

X = df["text"] # seperate features and label x- input text y- output class
y = df["label"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer( # uses top 1000 features
    max_features=1000,
    ngram_range=(1,2)    # considers unigrmas and bigrams
)

X = vectorizer.fit_transform(X).toarray() # transform text into numerical vectors

joblib.dump(vectorizer,"D:/mental-health-cnn-project/models/vectorizer.pkl") # save vectorizer so that same transformation is used in prediction

# encode labels
'''Depression → 0
Anxiety → 1
Bipolar → 2'''
encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(encoder,"D:/mental-health-cnn-project/models/label_encoder.pkl") # save encoder

# train test split
X_train,X_test,y_train,y_test = train_test_split( # 80 percent training and 20 percent testing
    X,y,test_size=0.2,random_state=42
)

feature_count = X_train.shape[1]

''' Model architecture'''
''' Input layer 1000 features , then dense layer 128 features and dense layer another 64'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(feature_count,)), # maximum features count
    tf.keras.layers.Dense(128,activation="relu"),  # learning complex patterns and relu is removing negatives
    tf.keras.layers.Dense(64,activation="relu"), # feature refining
    tf.keras.layers.Dense(len(set(y)),activation="softmax") # softmax gives probability distribution
])

model.compile(
    optimizer="adam", # optimizer adam for fast updation of neural network parameters to minimize loss function
    loss="sparse_categorical_crossentropy", # loss function used
    metrics=["accuracy"] # performance metric
)

''' model learns from pattern , adjusts weights using backpropagation'''
model.fit(
    X_train,
    y_train,
    epochs=10, # one complete pass in training
    batch_size=32,
    validation_data=(X_test,y_test)

)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

model.save("D:/mental-health-cnn-project/models/cnn_model.h5") # save the model and load the same file in prediction

print("Model trained successfully")