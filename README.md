# Quick overview

This repo contains the code to reproduce the results presented in the project, as well as the leaderboard submission. 

# Organization

## Data

The datasets are present in the _data_ subfolder.

## Main files

This repo includes the mandatory train et predict files. The _predict.py_ may be run without first running _train.py_, as the pre-trained model is already saved. _fun.py_ and _prep.py_ define functions to be used by the main files, and the _config.py_ file contains hyperparameters, so they must not be modified.

## Additional files and folders

In addition, we include:

-a _validate.py_ that trains the model on a split training set and generates in a reproducible fashion, numerical and graphical results, the latter being saved in the val_visualization folder.
-a folder called _gam_visualization_ containing the figures as well as a notebook producing those same figures.
-two folder, called _model_ and _scalers_, containing respectively the trained neural network and the scalers used during the preprocessing phase.

## Prediction

Running the _predict.py_ will result in the prediciton being saved in csv and zip formats under the _pred_ folder.
