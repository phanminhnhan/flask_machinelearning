# for ml 
import json
import os
from gensim.models import KeyedVectors 
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas,  numpy,  string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.layers import *
import gensim  
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# for api

from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)


# @app.route('/', methods=['POST'])
# def Home():
#     text = {"text": request.json["text"]}
#     X_data = pickle.load(open('X_data.pkl', 'rb'))
#     y_data = pickle.load(open('y_data.pkl', 'rb'))
    
#     tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
#     tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
#     X_data_tfidf =  tfidf_vect.transform(X_data)

#     svd = TruncatedSVD(n_components=300, random_state=42)
#     svd.fit(X_data_tfidf)

#     X_data_tfidf_svd = svd.transform(X_data_tfidf)

#     encoder = preprocessing.LabelEncoder()
#     y_data_n = encoder.fit_transform(y_data)

#     def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):       
#         X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
        
#         if is_neuralnet:
#             classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
            
#             val_predictions = classifier.predict(X_val)
#             test_predictions = classifier.predict(X_test)
#             val_predictions = val_predictions.argmax(axis=-1)
#     #         test_predictions = test_predictions.argmax(axis=-1)
#         else:
#             classifier.fit(X_train, y_train)
        
#             train_predictions = classifier.predict(X_train)
#             val_predictions = classifier.predict(X_val)
#     #         test_predictions = classifier.predict(X_test)
            
#         print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
#     #     print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


#     model = naive_bayes.MultinomialNB()
#     train_model(model, X_data_tfidf, y_data, is_neuralnet=False)

#     doc = preprocessing_doc(text)
#     doc_tfidf = tfidf_vect.transform([doc])
#     result = model.predict(test_doc_tfidf)
#     return jsonify(result[0])



def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines

class Ml(Resource):
    
    def post(self):   
        data = request.get_json(force=True)
        print(data["data"])
    
        text = data["data"]
        X_data = pickle.load(open('X_data.pkl', 'rb'))
        y_data = pickle.load(open('y_data.pkl', 'rb'))
        
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
        tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
        X_data_tfidf =  tfidf_vect.transform(X_data)

        svd = TruncatedSVD(n_components=300, random_state=42)
        svd.fit(X_data_tfidf)

        X_data_tfidf_svd = svd.transform(X_data_tfidf)

        encoder = preprocessing.LabelEncoder()
        y_data_n = encoder.fit_transform(y_data)

        def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):       
            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
            
            if is_neuralnet:
                classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
                
                val_predictions = classifier.predict(X_val)
                test_predictions = classifier.predict(X_test)
                val_predictions = val_predictions.argmax(axis=-1)
        #         test_predictions = test_predictions.argmax(axis=-1)
            else:
                classifier.fit(X_train, y_train)
            
                train_predictions = classifier.predict(X_train)
                val_predictions = classifier.predict(X_val)
        #         test_predictions = classifier.predict(X_test)
                
            print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
        #     print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


        model = naive_bayes.MultinomialNB()
        train_model(model, X_data_tfidf, y_data, is_neuralnet=False)

        doc = preprocessing_doc(text)
        doc_tfidf = tfidf_vect.transform([doc])
        result = model.predict(doc_tfidf)    
        return jsonify(result[0])
        return jsonify(data)

api.add_resource(Ml, '/ml')

if __name__ == '__main__':
    app.run(debug=True)    