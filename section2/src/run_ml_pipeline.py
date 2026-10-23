"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "../../section1/out"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "../out"
        self.smoke_test = False     # True = quick test (20 patients, 1 epoch); False = real training

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
        # Smoke test: use only 20 patients and 1 epoch, so the whole pipeline runs in a few minutes
    if c.smoke_test:
        data = data[:20]
        c.n_epochs = 5
        print("SMOKE TEST MODE: 20 patients, 5 epoch")



    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    
    # a. shuffle the patinet numbers (0...259). fixed seed
    np.random.seed(42)
    keys = np.random.permutation(len(data))

    # b. sizes: 60% train, 20% val 20% test
    n_train = int(len(data) * 0.6)
    n_val = int(len(data) * 0.2)

    # c. cut the shuffled array into three parts and every patient is exatly One Part --> no leackage
    split["train"] = keys[:n_train]
    split["val"] = keys[n_train:n_train+n_val]
    split["test"] = keys[n_train+n_val:]

    print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")



    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

