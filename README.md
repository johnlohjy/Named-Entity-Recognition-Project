# NAMED ENTITY RECOGNITION ON MIT MOVIE QUERIES

This project uses a Bi-Directional LSTMs with a CRF Layer neural network model architecture to perform named entity recognition. The use case here is on MIT movie queries where entities such as the actor and genre of the movie can be extracted from the query. 
For instance the movie query 

## Dependencies
* miniconda3 windows 64-bit environment
* visual c++ build tools
* jupyter notebook
* pandas 0.24.2
* numpy 1.18.1
* scikit-learn 0.21.2
* keras 2.2.4
* keras-contrib
* tensorflow 1.13.1
* nltk 3.4.4
* requests 2.22.0
* flask 1.1.1
* fasttext
* fasttext model

## Setup & Installation guide

1. Create a conda environment with python version 3.7.5. Example: <br> `conda create -n myenv python=3.7.5`
2. Download visual c++ build tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/. The installation may take a while.
3. Install pandas using `conda install -c anaconda pandas=0.24.2`
4. Download the fasttext master folder from https://github.com/facebookresearch/fastText
and add it inside the project folder. Run the `setup.py` file in the `fastText-master` folder <br> using `python setup.py install`
5. Install the packages below

Package | Command
------- | -------
jupyter notebook| `conda install -c anaconda jupyter`
scikit-learn 0.21.2| `conda install -c anaconda scikit-learn=0.21.2` 
keras 2.2.4| `conda install -c conda-forge keras=2.2.4` 
nltk 3.4.4| `conda install -c anaconda nltk=3.4.4`
requests 2.22.0| `conda install -c anaconda requests=2.22.0`
flask 1.1.1| `conda install -c anaconda flask=1.1.1`

6. Download the keras-contrib master folder from https://github.com/keras-team/keras-contrib and add it inside the project folder. Run the `setup.py` file in the `keras-contrib-master` folder using<br>  `python setup.py install`

7. Download the `English bin+text` from https://fasttext.cc/docs/en/pretrained-vectors.html and add the downloaded `wiki.en.bin` file to the project folder: `ner_backend`

## Explanation of Project files
NOTE: *For more code depth, please refer to the comments and markdown annotations the notebooks.
The explanation for bulk of the code for `B_random_cv_mit_movie_query.ipynb` can be found in `C_trainer.ipynb`. And the explanation for the bulk of the code for `E_mass_predictor.ipynb` can be found in `D_predictor.ipynb`. And, the explanation for the use of Bi-Directional LSTM neural networks with a CRF layer can be found at the bottom of the `C_trainer.ipynb` notebook. Thank you!*

The project mainly consists of 5 files
1. `A_mit_movie_queries_dataset_conversion.ipynb`
2. `B_random_cv_mit_movie_query.ipynb`
3. `C_trainer.ipynb`
4. `D_predictor.ipynb`
5. `E_mass_predictor.ipynb`

The order in which the files are named alphabetically is also the sequence in which they should be ran when training your own model. However, I have already done that so that you do not need to. Before running files labelled B to E, ensure you have ran the `app_word_vector.py` file, which is found in the same directory as the rest of the files. Instructions to run it can be found under the sub-heading `app_word_vector.py`.

The files have comments and is well documented, making it easier for you to follow. 

#### `A_mit_movie_queries_dataset_conversion.ipynb`

This file is used to scrape the movie queries training data from MITs website and output a preprocessed training dataset, along with other files and can be done so by running all cells. The output from running this file includes the following:
* `movie_queries.txt` file - this pickled file contains the raw training dataset scraped from the MIT website. It can be found in the `training_set` folder.
* `movie_queries_training_dataset.csv` file - this csv file contains the contains the pre-processed dataset for training the neural network model. It can be found in the `training_set` folder.
* `target_to_index.txt` file - this pickled file contains a dictionary with the keys as the targets and the values as the indexes. This is helpful for converting the targets of the training dataset to indexes as neural network models accept vector inputs. It can be found in the `index_converter` folder.
* `index_to_target.txt` file - this pickled file contains a dictionary with the keys as the indexes and the values as the targets. This is helpful for converting the indexes of the predicted array to targets during prediction. It can be found in the `index_converter` folder.


#### `B_random_cv_mit_movie_query.ipynb`

*Do not run the `B_random_cv_mit_movie_query.ipynb` and `C_trainer.ipynb` files at the same time, as it is not recommended to have > 1 heavy tensorflow process running at the same time such as training or performing random search cross validation.*

This file is used to perform random search cross validation, to find the best hyperparameters that will allow the named entity recognition, neural network model to best learn from the use case. If this is your first time running the file, or you do not wish to continue running the random search cross validation from where you last left off, run all cells but __do not run the `Continue random cv` code block__. However, please note that the random search cross validation may take a few days as performing random search cross validation on 60 different, random combinations hyperparameters (optimal number, explanation can be found in the notebook) takes quite abit of time. 

Do not worry about stopping the program or any unforseen circumstances which may terminate the program as the progress of the random search cross validation is constantly saved after each successful cross validation of a combination of hyperparameter, which will allow you to continue running from where you left off by running the `Continue random cv` code block.
__If you do want to continue from where you left off, run all cells, including the `Continue random cv` code block, but do not run the `Generating Hyperparameters` code block__ as this will overwrite the previous list of combinations of hyperparameters that you were supposed to test. 

If the `Find best parameters` code block is ran, a pickled `best_parameter_info` txt file, which is a list containing the best combination of hyperparameters and its respective average f1 score attained from the random search cross validation process, will be saved. This `best_parameter_info` txt file is helpful in training the model as it provides the best combination of hyperparameters. 

In summary, the output from running this file includes the following:
* `random_search_hyperparams.txt` file - this pickled file contains the list of combination of hyperparameters that has been/is to be tested for random cross validation, depending on where you have stopped.
* `random_search_hist.txt` file - this pickled file contains a dictionary where each key is the index number of the combination of the hyperparameters in the `random_search_hyperparams` list and each value is a list, containing information regarding the combination of hyperparameters tested, the mean f1 score of all the 5 folds that were tested during cross validation, as well as all the f1 scores of the 5 folds that were tested during cross validation. This is the progress of the random search cross validation you have ran.
* `best_hyperparameter_info.txt` file - this pickled file contains a list that has the best combination of hyperparameters and its respective average f1 score attained from the random search cross validation process. Once again, this is helpful in training the model as it provides the best combination of hyperparameters. 

However, please note that these files are not provided in this repository.


#### `C_trainer.ipynb`

*Do not run the `B_random_cv_mit_movie_query.ipynb` and `C_trainer.ipynb` files at the same time, as it is not recommended to have > 1 heavy tensorflow process running at the same time such as training or performing random search cross validation.*

This file is used to train the named entity recognition, neural network model and can be done so by running all cells. The best hyperparameter info under the `Getting best hyperparameters of model` code block is the best result from my own random search cross validation. If you wish to use your own best hyperparameters found from your own random search cross validation, please change the cell to a markdown cell and change the cell below to a code cell - it will extract the hyperparameters from your `best_hyperparameter_info.txt` file and use that to train the model. Only the best weights from training is left in the `model_training_weights` folder, the rest of the weights that have been saved from each epoch are deleted.

To add on, the explanation for the use of Bi-Directional LSTM neural networks with a CRF layer can be found at the bottom of the notebook.

The output from this file includes the following:
* `best model weight.hdf5` - the best model weights are loaded to do prediction in the `D_predictor.ipynb` and `E_mass_predictor.ipynb` files. This can be found in the `model training weights` folder.
* `f1_hist.txt` - this file has the f1 scores of all epochs during training. This can be found in the `training hist` folder.

#### `D_predictor.ipynb`
This file is used to predict a single movie search query where the results can be found at the bottom of the notebook and can be done so by running all cells. Under the `Input text to be extracted` code block, I have used test texts that I extracted to a list. If you wish to test using your own text, you can simply change that cell to a markdown cell and the cell below it to a code cell and edit accordingly.

#### `E_mass_predictor.ipynb`
This file is used to predict multiple movie search queries stored in a list, where the results are saved to the `mass_predictor_results` folder as csv files, it can done by running all cells. If you wish to test using your own set of texts, you can simply change that cell to a markdown cell and the cell below it to a code cell and edit accordingly.

The output from this file is the csv files of the predicted results of each text which can be found in the `mass_predictor_results` folder.


### Additional files:
#### `AA_mit_movie_queries_dataset_conversion_testset.ipynb`
This file is used to output test data and can be done by running all cells. 
The output from this file includes the following: 
* `movie_queries_test.txt` file - this pickled file contains the raw test dataset scraped from the MIT website. It can be found in the `test_set` folder.
* `movie_queries_test_text.txt` file - this pickled file contains a list of movie queries in sentence form, which can be used to test prediction. It can be found in the `test_set` folder.
* `movie_queries_test_text_targets.txt` file - this pickled file contains a list of the movie queries targets, which can be used for checking the performance of the prediction. It can be found in the `test_set` folder.

#### `app_word_vector.py`
This file/API is used to convert the tokenized text into vectors. This is needed because neural network mdoels cannot read raw text and need vector inputs. To run the API:
1. Change directory to the project folder
2. Type `SET FLASK_APP=app_word_vector.py` and hit enter
3. Type `FLASK RUN` and hit enter
4. Wait for the API to run before running notebooks labelled from B to E








## Brief Explanation of Folders

#### index_converter folder 
This folder contanins 2 txt files. The `target_to_index` txt file, as its name suggests, helps convert target labels to their index form. It is a pickled dictionary with the keys as the targets and the values as the indexes. The `index_to_target` txt file, as its name suggests, helps convert indexes to their corresponding target labels. It is a pickled dictionary with the keys as the indexes and the values as the targets.

 #### mass_predictor_results folder 
 
 This folder contains the csv files of the predicted results of each text, when the `E_mass_predictor` file is ran on multiple test texts.

 #### model_training_weights folder
 This folder contains the weights of all epochs during training. However, a program is ran such that only the best performing model weight is left in the folder and the rest are deleted.

 #### random_search_data folder
 This folder will contain 3 txt files if you run the `B_random_cv_mit_movie_query` file. The pickled `random_search_params` txt file contains the list of combination of hyperparameters that has been/is to be tested for random cross validation, depending on where you have stopped. The pickled `random_search_hist` txt file contains a dictionary where each key is the index number of the combination of the hyperparameters in the `random_search_params` list and each value is a list, containing the combination of hyperparameters tested, the mean f1 score of all the 5 folds that were tested during cross validation, as well as all the f1 scores of the 5 folds that were tested during cross validation. This is the progress of the random search cross validation you have ran. The pickled `best_parameter_info` txt file is a list containing the best combination of hyperparameters and its respective average f1 score attained from the random search cross validation process. 

 #### test_set folder
 This folder contains 3 txt files. The pickled `movies_queries_test` txt file contains the raw test dataset scraped from the MIT website. The pickled `movie_queries_test_text` txt file contains the movie queries in sentence form. The pickled `movie_queries_test_text_targets` contains the targets of the movie queries sentences.

 #### training_hist folder
 This folder contains the pickled `f1_hist` txt file that has the f1 scores of all epochs during training.

 #### training_set folder
 This folder contains 1 txt file and 1 csv file. The pickled `movie_queries` txt file contains the raw training dataset scraped from the MIT website. The `movie_queries_training_dataset` csv file contains the pre-processed dataset for training.

 #### word_vector folder
 This folder contains 1 txt file, the pickled `word_vectors` txt file that contains a list of word vectors that the API has processed from the words it recieved 



