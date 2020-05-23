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
#from flask_cors import CORS

#API definition, create the flask app
app = Flask(__name__)
#cors = CORS(app)

ft = fasttext.load_model(fasttext_model_path)
fasttext.util.reduce_model(ft,50)

#Check if the word is a number
def containsNumbers(check):
    return any(char.isdigit() for char in check)


@app.route("/word_vectorization", methods=['POST'])
def word_vectorization():
    word_vectors = []
    tokenized_text_lower = request.json
    for word in tokenized_text_lower:
        if containsNumbers(word):
            word_vector = ft.get_word_vector('<NUMBER>')
            word_vectors.append(word_vector)
            continue

        word_vector = ft.get_word_vector(word)

        word_vectors.append(word_vector)

    with open(word_vectors_filepath, "wb") as t:
        pickle.dump(word_vectors, t)
    return Response(status = 200)

if __name__ == '__main__':
    app.run(debug=True)