# aps360project
APS360 Group 41

# Project files and folders
- LieWaves/
    - raw data downloaded directly from the internet, do not use directly
- preprocessing.py
    - python script to extract raw data and perform some pre-processing on it so that it can be fed to the neural net
    - creates n x 5 x 90 x 1 input tensors, where n depends on whether it is the training set, test set, or validation set
- ProcessedInputData/
    - contains the pre-processed data to be fed into the neural net
    - data is stored as .npy arrays which can and should be imported directly
    - x is the inputs while y is the corresponding labels
- model1.ipynb
    - example model performing data imports and some basic training
    - the linear model here is not able to fit the data but the shapes and sizes used are able to properly demonstrate use of BCE loss function

