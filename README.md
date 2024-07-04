# KilterGPT

Repo for training a model that generates Kilterboard climbs.

## Data

One thing you need to do is download the latest version of the kilterboard apk file, unzip it, and copy the `db.sqlite3` file to the `data` directory.
Then run the `preprocessing.ipynb` notebook to generate the data for training.


## Training

Simply running the `train.py` script will start training the model. You can adjust the hyperparameters in the script.
