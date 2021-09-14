import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import utils

df = pd.read_csv('dataset/fer2013.csv')

emotion_label = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

INTERESTED_LABELS = [0, 2, 3, 4, 5, 6]
df = df[df.emotion.isin(INTERESTED_LABELS)]
 
#Balance data
file_count = 4002
samples = []
for category in df['emotion'].unique():
    category_slice = df.query("emotion == @category")
    samples.append(category_slice.sample(file_count, replace=False, random_state=1))
df = pd.concat(samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)

le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = utils.to_categorical(img_labels)

X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=42)

#Normalizing data
X_train = X_train/255
Y_train = X_train/255
 

##Build model
