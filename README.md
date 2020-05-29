# NAMED ENTITY RECOGNITION ON MIT MOVIE QUERIES
### Please note that this project has been adapted for the use-case of performing Named Entity Recognition on MIT movie queries as the data for the original use case that this project was used for (done with a leading investment organisation in Singapore) is highly sensitive and cannot be publicly disclosed. Thus, the Named Entity Recognition code of this project will differ from the Named Entity Recognition backend code of the original school major project which won the ACI Singapore - The Financial Markets Association Project Prize. For instance, different and less features in the training set. However, the methodology and code is largely the same except for a different dataset and some tweaks to this new use case.

## Please read this documentation to understand the project better.

This project uses a Bi-Directional LSTM with CRF Layer neural network model architecture to perform named entity recognition. The use case here is on MIT movie queries where entities such as the actor and genre of the movie can be extracted from the movie query. For instance the movie query "show me 1980s action movies" will have entities such as year which would be "1980s" and genre which would be "action".

## Dependencies
* miniconda3 windows 64-bit environment
* visual c++ build tools
* jupyter notebook
* pandas 0.24.2
* scikit-learn 0.21.2
* keras 2.2.4 (tensorflow backend)
* keras-contrib
* nltk 3.4.4
* requests 2.22.0
* flask 1.1.1
* fasttext
* fasttext model

## Setup & Installation guide

1. Create a conda environment with python version 3.7.5. Example: <br> `conda create -n myenv python=3.7.5`
2. Download and install visual c++ build tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/. The installation may take a while.
3. Install pandas using `conda install -c anaconda pandas=0.24.2`
4. Download the fasttext master folder from https://github.com/facebookresearch/fastText
and add the fasttext master folder inside the project folder. Run the `setup.py` file in the `fastText-master` folder <br> using `python setup.py install`
5. Install the packages below

Package | Command
------- | -------
jupyter notebook| `conda install -c anaconda jupyter`
scikit-learn 0.21.2| `conda install -c anaconda scikit-learn=0.21.2` 
keras 2.2.4| `conda install -c conda-forge keras=2.2.4` 
nltk 3.4.4| `conda install -c anaconda nltk=3.4.4`
requests 2.22.0| `conda install -c anaconda requests=2.22.0`
flask 1.1.1| `conda install -c anaconda flask=1.1.1`

6. Download the keras-contrib master folder from https://github.com/keras-team/keras-contrib and add the keras-contrib master folder inside the project folder. Run the `setup.py` file in the `keras-contrib-master` folder using<br>  `python setup.py install`

7. Download the `English bin+text` from https://fasttext.cc/docs/en/pretrained-vectors.html and add the downloaded `wiki.en.bin` file to the project folder. The installation may take a while.

Your project folder structure should look like this:

|__fastText-master <br>
|__index_converter <br>
|__keras-contrib-master <br>
|__mass_predictor_results <br>
|__model_training_weights <br>
|__random_search_data <br>
|__test_set <br>
|__training_hist <br>
|__training_set <br>
|__word_vector <br>
|__A_mit_movie_queries_dataset_conversion.ipynb <br>
|__AA_mit_movie_queries_dataset_conversion_testset.ipynb <br>
|__app_word_vector.py <br>
|__B_random_cv_mit_movie_query.ipynb <br>
|__C_trainer.ipynb <br>
|__D_predictor.ipynb <br>
|__E_mass_predictor.ipynb <br>
|__wiki.en.bin <br>
<br>
<br>

## Explanation of Project files
NOTE: For more code depth, please refer to the markdown annotations and comments in the notebooks.
* The explanation for bulk of the code for `B_random_cv_mit_movie_query.ipynb` can be found in `C_trainer.ipynb`. 
* The explanation for the bulk of the code for `E_mass_predictor.ipynb` can be found in `D_predictor.ipynb`. 
* The explanation for the use of Bi-Directional LSTM neural networks with a CRF layer can be found at the bottom of the `C_trainer.ipynb` notebook. 
Thank you!

<br>

The project consists of mainly 6 files
1. `app_word_vector.py`
2. `A_mit_movie_queries_dataset_conversion.ipynb`
3. `B_random_cv_mit_movie_query.ipynb`
4. `C_trainer.ipynb`
5. `D_predictor.ipynb`
6. `E_mass_predictor.ipynb`

The order in which the files are named alphabetically is also the sequence in which they should be ran when training your own model. However, I have already done that so that you do not need to, you can jump straight to the `D_predictor.ipynb` or `E_mass_predictor.ipynb` file to test the model. 

__Before running files labelled B to E__, ensure you have ran the `app_word_vector.py` file, which is found in the same directory as the rest of the files. Instructions to run it can be found under the sub-heading app_word_vector.py.

The files all have comments and are well documented, making it easier for you to follow. 

<br>

#### `app_word_vector.py` 

This file/API is used to convert tokenized text into vectors and is called by notebooks that require this API. This conversion is needed because neural network mdoels cannot read raw text and need vector inputs. To run the API:
1. Change directory to the project folder in your conda environment
2. Type `SET FLASK_APP=app_word_vector.py` and hit enter
3. Type `FLASK RUN` and hit enter
4. Wait for the API to run before running notebooks labelled from B to E. Note that this may take a while

When this API is called, the output is a pickled `word_vector.txt` file that contains a list of word vectors that the API has converted from the words it recieved. This file can be found in the word_vector folder.

<br>

#### `A_mit_movie_queries_dataset_conversion.ipynb`

This file is used to scrape the raw movie queries training data from MITs website and output a preprocessed training dataset, along with other files. This process can be done by running all cells in this notebook. The output from running this file includes the following:
* `movie_queries.txt` file - this pickled file contains the raw training data scraped from the MIT website. It can be found in the training_set folder.
* `movie_queries_training_dataset.csv` file - this csv file contains the contains the pre-processed training dataset for training the neural network model. It can be found in the training_set folder.
* `target_to_index.txt` file - this pickled file contains a dictionary with each key as a target label and each value as an unique index. This is helpful for converting the target labels of the pre-processed training dataset to indexes as neural network models only accept vector inputs. It can be found in the index_converter folder.
* `index_to_target.txt` file - this pickled file contains a dictionary with each key as an unique indexe and each value as a target label. This is helpful for converting the indexes of the predicted array to target labels during prediction. It can be found in the index_converter folder.

<br>

#### `B_random_cv_mit_movie_query.ipynb`

*Do not run the `B_random_cv_mit_movie_query.ipynb` and `C_trainer.ipynb` files at the same time, as it is not recommended to have > 1 heavy tensorflow process running at the same time such as training or performing random search cross validation.*

This file is used to perform random search cross validation, to find the best hyperparameters that will allow the neural network model to best learn from the use case. 

__If this is your first time running the file, or you do not wish to continue running the random search cross validation from where you last left off__, run all cells in this file except for the "Continue random cv" code block. Please note that the random search cross validation may take a few days as performing random search cross validation on 60 different, random combinations hyperparameters (optimal number, explanation can be found in the notebook) takes quite abit of time. 

Do not worry about stopping the program or any unforseen circumstances which may terminate the program as the progress of the random search cross validation is constantly saved after each successful cross validation of a combination of hyperparameter, which will allow you to continue running from where you left off by running the `Continue random cv` code block.

__If you do want to continue from where you left off, run all cells, including the "Continue random cv" code block, but do not run the Generating Hyperparameters code block__ as this will overwrite the previous list of combinations of hyperparameters that you were supposed to test. 


In summary, the output from running this file includes the following:
* `random_search_hyperparams.txt` file - this pickled file contains the list of combination of hyperparameters that has been/is to be tested for random cross validation
* `random_search_hist.txt` file - this pickled file contains a dictionary where each key is the index number of the combination of the hyperparameters in the random_search_hyperparams.txt list and each value is a list, containing information regarding the combination of hyperparameters tested, the mean f1 score of all the 5 folds that were tested during cross validation, as well as all the f1 scores of the 5 folds that were tested during cross validation. This is the progress of the random search cross validation you have ran.
* `best_hyperparameter_info.txt` file - this pickled file contains a list that has the best combination of hyperparameters and its respective average f1 score attained from the random search cross validation process. This is helpful in training the model and for prediction as it provides the best combination of hyperparameters for our use case. 

<br>

#### `C_trainer.ipynb`

*Do not run the `B_random_cv_mit_movie_query.ipynb` and `C_trainer.ipynb` files at the same time, as it is not recommended to have > 1 heavy tensorflow process running at the same time such as training or performing random search cross validation.*

This file is used to train the neural network model. Only the best weights from training is left in the model_training_weights folder, the rest of the weights that have been saved from each epoch are deleted. The training can be done by running all cells in this notebook. 

__If you wish to use your own best hyperparameters found from your own random search cross validation__, please make sure you have ran the `B_random_cv_mit_movie_query.ipynb` notebook and have your `best_hyperparameter_info.txt` file. Otherwise, my best combination of hyperparameters found from my own random search cross validation will be used.

Also, the explanation for the use of Bi-Directional LSTM neural networks with a CRF layer can be found at the bottom of the notebook.

The output from this file includes the following:
* `weights.best.hdf5` - the best model weights left from training that are loaded to do prediction in the `D_predictor.ipynb` and `E_mass_predictor.ipynb` files. This can be found in the model training weights folder.
* `f1_hist.txt` - this file has the f1 scores of all epochs during training. This can be found in the training hist folder.

<br>

#### `D_predictor.ipynb`
This file is used to predict a single movie search query where the results can be found at the bottom of the notebook. The prediction can be done by running all cells in this notebook. 

__If you wish to predict your own movie query__, please edit the "Input text to be extracted" code block accordingly. Instructions to make this change can be found in its annotations. 

__If you wish to use your own best hyperparameters found from your own random search cross validation__, please make sure you have ran the `B_random_cv_mit_movie_query.ipynb` notebook and have your `best_hyperparameter_info.txt` file. Otherwise, my best combination of hyperparameters found from my own random search cross validation will be used.

<br> 

#### `E_mass_predictor.ipynb`
This file is used to predict multiple movie search queries stored in a list, where the results are saved to the mass_predictor_results folder as csv files. The mass prediciton can done by running all cells in this notebook. 

__If you wish to predict your own list of movie queries__, please edit the "Input text to be extracted" code block accordingly. Instructions to make this change can be found in its annotations. 

__If you wish to use your own best hyperparameters found from your own random search cross validation__, please make sure you have ran the `B_random_cv_mit_movie_query.ipynb` notebook and have your `best_hyperparameter_info.txt` file. Otherwise, my best combination of hyperparameters found from my own random search cross validation will be used.

The output from this file is the csv files of the predicted results of each text which can be found in the mass_predictor_results folder.

<br>

### Additional file:
#### `AA_mit_movie_queries_dataset_conversion_testset.ipynb`
This file is used to output test data and can be done by running all cells. 
The output from this file includes the following: 
* `movie_queries_test.txt` file - this pickled file contains the raw test dataset scraped from the MIT website. It can be found in the test_set folder.
* `movie_queries_test_text.txt` file - this pickled file contains a list of movie queries in sentence form, which can be used to test prediction. It can be found in the test_set folder.
* `movie_queries_test_text_targets.txt` file - this pickled file contains a list of the movie queries targets, which can be used for cross-checking the performance of the prediction. It can be found in the test_set folder.


<br>
<br>


## Brief Explanation of Folders

#### index_converter folder 
This folder contanins 2 files. 

1. `target_to_index.txt` file - this pickled file contains a dictionary with each key as a target label and each value as an unique index. This is helpful for converting the target labels of the pre-processed training dataset to indexes as neural network models only accept vector inputs. 
2. `index_to_target.txt` file - this pickled file contains a dictionary with each key as an unique indexe and each value as a target label. This is helpful for converting the indexes of the predicted array to target labels during prediction.

 #### mass_predictor_results folder 
 
 This folder contains the csv files of the predicted results of each text, when the `E_mass_predictor` file is ran on multiple test texts.

 #### model_training_weights folder
 This folder contains the weights of all epochs during training. However, code is ran such that only the best performing model weights is left in the folder and the rest are deleted i.e. `weights.best.hdf5`.

 #### random_search_data folder
 This folder will contain 3 files if you run the `B_random_cv_mit_movie_query` file. If not, it is empty. 
1. `random_search_params.txt` - this pickled file contains the list of combination of hyperparameters that has been/is to be tested for random cross validation.
2. `random_search_hist.txt` - this pickled file contains a dictionary where each key is the index number of the combination of the hyperparameters in the random_search_hyperparams.txt list and each value is a list, containing information regarding the combination of hyperparameters tested, the mean f1 score of all the 5 folds that were tested during cross validation, as well as all the f1 scores of the 5 folds that were tested during cross validation. This is the progress of the random search cross validation you have ran.
3. `best_parameter_info.txt`  - this pickled file contains a list that has the best combination of hyperparameters and its respective average f1 score attained from the random search cross validation process. This is helpful in training the model and for prediction as it provides the best combination of hyperparameters for our use case for the notebooks to use.  

 #### test_set folder
 This folder contains 3 files. 
 1. `movies_queries_test.txt` - this pickled file contains the raw test dataset scraped from the MIT website.
 2. `movie_queries_test_text.txt` - this pickled file contains a list of movie queries in sentence form, which can be used to test prediction.
 3. `movie_queries_test_text_targets.txt` - this pickled file contains a list of the movie queries targets, which can be used for cross-checking the performance of the prediction.

 #### training_hist folder
 This folder contains the pickled `f1_hist.txt` file that has the f1 scores of all epochs during training.

 #### training_set folder
 This folder contains 2 files. 
`movie_queries.txt` - this pickled file contains the raw training data scraped from the MIT website.
`movie_queries_training_dataset.txt` - this csv file contains the contains the pre-processed training dataset for training the neural network model. It can be found in the training_set folder.

 #### word_vector folder
 This folder contains 1 txt file, the pickled `word_vectors` txt file that contains a list of word vectors that the Word Vectorization API has processed from the words it recieved.



