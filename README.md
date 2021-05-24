# Mining-and-Relating-Design-Contexts-and-DesignPatterns-from-Stack-Overflow

# DPC Miner

## How to use DPC Miner

There are two ways you can use the DPC Miner on the data

###### Use the trained RL-Model

You can use the already trained RL-Model on DPC Miner to mine for more data using DPC_Miner.Py

Step 1: Place the file you want to mine as "read_miner" / (Give the location of 'More_SO_data.csv')

Step 2: Place the address of labelled dataset as "read_classifier_data" / (Give the location of 'Classifier_data.csv')

Step 3: Load the trained RL_Model (RL_Model.model)

Step 4: Run DPC_Miner.py
- Output: mining predictions

###### Train both RL_Model and DPC_Miner
Step 1: In RL_Model_Training.py
- Place the file you want to train the RL_Model as "read_file_raw" / (Give the location of 'Sample_for_RLTraining.csvv')

Step 2: Run RL_Model_Training.py
- Output: RL_Model.model

Step 3: Run DCP_Miner.py 



## Data Files

- More_SO_data.csv -- Contains the posts that need to predict as DPC or not. 
- Classifier_data.csv -- Contains the posts that are classified and need to train the DPC_Miner.
- Sample_for_RLTraining.csv -- Contains sample posts that can use to train the RL_Model. 
- Labelled_Data.csv -- Labelled DPC dataset

