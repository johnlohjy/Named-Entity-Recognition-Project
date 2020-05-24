import os
curr_dir = os.getcwd()

#Filepaths
fasttext_model_path = os.path.join(curr_dir,'wiki.en.bin').replace('\\','/')
word_vectors_filepath = os.path.join(curr_dir,'word_vector','word_vector.txt').replace('\\','/')



#Dependencies
import requests
import pickle
import fasttext.util
from flask import Flask, request, jsonify, Response

#API definition, create the flask app
app = Flask(__name__)

#Load fasttext model
ft = fasttext.load_model(fasttext_model_path)
#Reduce the dimensionality of word vectors
fasttext.util.reduce_model(ft,50)

#Check if the word is a number
def containsNumbers(check):
    return any(char.isdigit() for char in check)


@app.route("/word_vectorization", methods=['POST'])
def word_vectorization():
    #Initialize empty word_vectors list
    word_vectors = []
    #Retrieve the list of words
    tokenized_text_lower = request.json
    #Iterate over the list of words
    for word in tokenized_text_lower:
        #Standardise vector for numbers as they have no semantic meaning
        if containsNumbers(word):
            word_vector = ft.get_word_vector('<NUMBER>')
            word_vectors.append(word_vector)
            continue

        #Get the word vector for the word
        word_vector = ft.get_word_vector(word)

        #Append the word vector to the word_vectors list
        word_vectors.append(word_vector)

    #Dump the word vectors list
    with open(word_vectors_filepath, "wb") as t:
        pickle.dump(word_vectors, t)
    return Response(status = 200)

if __name__ == '__main__':
    app.run(debug=True)