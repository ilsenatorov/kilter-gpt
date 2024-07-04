# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Installation

`pip3 install torch torchvision torchaudio wandb lightning plotly pandas numpy matplotlib ipykernel jupyter`

## Data

One thing you need to do is download the latest version of the kilterboard apk file, unzip it, and copy the `db.sqlite3` file to the `data` directory.
Then run the `preprocessing.ipynb` notebook to generate the data for training.


## Training

Simply running the `train.py` script will start training the model. You can adjust the hyperparameters in the script.

## TODO

* Save and load tokenizer from json/pickle
* Improve tokenizer functionality, move all the tokenization logic to the tokenizer class
* Improve consistency of data preprocessing store as pandas dataframe, handle internally as list/tensor
* Better config handling
* Validation set and loop?
* Tokenize angle/difficulty and add them to the model
* Add some automatic evaluation metrics (similarity to real data, consistency, etc.)
* Log the model to wandb
* Add a script/notebook to generate climbs from the model
* Convert dataset to hf dataset?
* Add a script to convert the model to onnx?
* Plot UMAP of token embeddings to check for logical clustering
