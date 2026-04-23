# CS5180 Final Project - Model-Free vs. Model-Based Reinforcement Learning in Atari Environments with Limited Experience
Final Project for CS5180. Using Rainbow DQN and SimPLe Algorithms in Atari Environments. William Cohen and Sathvik Charugundla 

We ran our code in Google Colab using A1000 GPU High RAM machines. For our implementation of the Rainbow DQN, we used Google Dopamine (https://github.com/google/dopamine), starting with the hyperparameters they recommended, then modified them to see if our results could improve. This repository hasn't been updated in some years so in the colab, we needed to install some older versions of some required libraries. We also needed to create a folder structure baselines/common/ and place atari_wrappers.py and wrappers.py from https://github.com/openai/baselines/tree/master/baselines/common in order for the code to compile each time we connected to a runtime environment.

For SimPLe, we used Thomas Schillaci's SimPLe repo (https://github.com/thomas-schillaci/SimPLe). It wasn't maintained, so I refactored the code with updated dependencies and convinience functions used in training (refer to SimPLe/UPDATES.md).
