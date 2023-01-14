# IMPORTS
from flask import Flask
from joblib import dump, load
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# path to dataset
data = 'Star_Classifier_Data/Stars.csv'

# load data
stars = pd.read_csv(data)

star_types = {
        0: "Red Dwarf",
        1: "Brown Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Super Giant",
        5: "Hyper Giant"
    }

# change star types from integers to real types of stars
for index in tqdm(range(len(star_types.keys()))):
    stars["Type"] = stars["Type"].replace(index, star_types[index])


# !!! TO BE REMOVED UPON IMPLEMENTING OF KEYBOARD INPUT !!!
# get test data
nr_test_values = 5

test_values = stars.sample(nr_test_values)

test_values_X = test_values.drop(["Type", "Color", "Spectral_Class"], axis=1)
test_values_y = test_values["Type"]

# !!! TO BE REMOVED UPON IMPLEMENTING OF KEYBOARD INPUT !!!

# Load Star Classifier Model
clf = load('Star_Classifier_Model/Star_Classifier.joblib')

# Make predictions
predictions = clf.predict(test_values_X)

# Compute Accuracy Score
score = accuracy_score(test_values_y, predictions)


app = Flask(__name__)

@app.route('/')
def hello():
    # return list of predictions
    return predictions.tolist()

@app.route('/score')
def model_score():
    # return accuracy score
    return str(score)

@app.route('/y')
def y_values():
    # return list of y
    return test_values_y.tolist()
