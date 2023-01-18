# IMPORTS

from flask import Flask, redirect, url_for, render_template, request, jsonify
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

# Load Star Classifier Model
star_clf = load('Star_Classifier_Model/Star_Classifier.joblib')

# Load Neo Classifier Model
neo_clf = load('Neo_Classifier/Neo_Classifier.joblib')

# Get data from frontend


app = Flask(__name__)

'''
        id = request.json['id']
    name = request.json['name']
    estDiameterMin = request.json['estDiameterMin']
    estDiameterMax = request.json['estDiameterMax']
    relativeVelocity = request.json['relativeVelocity']
    missDistance = request.json['missDistance']
    orbitingObject = request.json['orbitingObject']
    sentryObject = request.json['sentryObject']
    absoluteMagnitude = request.json['absoluteMagnitude']
'''

@app.route('/neo', methods=["POST"])
def neo():
    id = {'id': request.json['id']}
    name = {'name': request.json['name']}
    estDiameterMin = {'estDiameterMin': request.json['estDiameterMin']}
    estDiameterMax = {'estDiameterMax': request.json['estDiameterMax']}
    relativeVelocity = {'relativeVelocity': request.json['relativeVelocity']}
    missDistance = {'missDistance': request.json['missDistance']}
    orbitingObject = {'orbitingObject': request.json['orbitingObject']}
    sentryObject = {'sentryObject': request.json['sentryObject']}
    absoluteMagnitude = {'absoluteMagnitude': request.json['absoluteMagnitude']}

    neo_data = [id['id'], name['name'], estDiameterMin['estDiameterMin'], estDiameterMax['estDiameterMax'], relativeVelocity['relativeVelocity'], missDistance['missDistance'],
            orbitingObject['orbitingObject'], sentryObject['sentryObject'], absoluteMagnitude['absoluteMagnitude']]

    print(neo_data)

    input_df = pd.DataFrame([neo_data], columns=['id', 'name', 'est_diameter_min', 'est_diameter_max',
                                           'relative_velocity', 'miss_distance', 'orbiting_body', 'sentry_object',
                                           'absolute_magnitude'])

    input_df = input_df.drop([ "sentry_object", "orbiting_body", "id", "name"], axis=1)

    neo_prediction = neo_clf.predict(input_df)

    return jsonify({'result': str(neo_prediction)})


@app.route('/stars', methods=["POST"])
def star():
    temperature = {'temperature': request.json['temperature']}
    relativeLuminosity = {'relativeLuminosity': request.json['relativeLuminosity']}
    relativeRadius = {'relativeRadius': request.json['relativeRadius']}
    absoluteMagnitude = {'absoluteMagnitude': request.json['absoluteMagnitude']}
    color = {'color': request.json['color']}
    spectralClass = {'spectralClass': request.json['spectralClass']}

    star_data = [temperature['temperature'],
                 relativeLuminosity['relativeLuminosity'],
                 relativeRadius['relativeRadius'],
                 absoluteMagnitude['absoluteMagnitude'],
                 color['color'],
                 spectralClass['spectralClass']]

    print(star_data)

    input_df = pd.DataFrame([star_data], columns=['Temperature',
                                                  'L',
                                                  'R',
                                                  'A_M',
                                                  'Color',
                                                  'Spectral_Class'])

    input_df = input_df.drop(["Color", "Spectral_Class"], axis=1)

    star_prediction = star_clf.predict(input_df)

    return jsonify({'result': str(star_prediction)})

























'''
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

# Make predictions
predictions = clf.predict(test_values_X)

# Compute Accuracy Score
score = accuracy_score(test_values_y, predictions)

'''
