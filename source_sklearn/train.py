from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--max_leaf_nodes', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=3)
    parser.add_argument('--min_samples_split', type=int, default=3)    

    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    
    max_leaf_nodes = args.max_leaf_nodes
    max_depth= args.max_depth
    min_samples_leaf = args.min_samples_leaf
    min_samples_split = args.min_samples_split

    ## TODO: Define a model 
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, 
                                   max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf, 
                                   min_samples_split=min_samples_split)
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    ## Get accuracy of prediction on the training dataset
    y_train_pred = model.predict(train_x)
                 
    ## Calculate metric and write to cloudwatch log
    train_accuracy = accuracy_score(train_y, y_train_pred)
    print('train-accuracy: {};'.format(train_accuracy))
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
