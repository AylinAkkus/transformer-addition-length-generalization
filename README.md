# progress-measures-paper

`transformers.py` contains the code to train the model. `Grokking_Analysis.ipynb` contains the code to load the saved checkpoints for the mainline run, calculate the progress metrics on it, and plots the figures. `Non_Modular_Addition_Grokking_Tasks.ipynb` contains training code for the non-modular addition experiments.

# To Do:
- For charcter level tokenization with varying lenghts (e.g. 12+31=) do an accuracy table for all cases (e.g. |operator 1|= |operator 2| = 1; |operator 1| = 1,  |operator 2| = 2) etc) on train and test
- Try character level tokenization with 0s to fix the varying length, do another accuracy table
- Add a seed() to fix the train and test set
- Vary positional encodings (current implementations learns the positional encodings)
- Change folder structure and save the config for different runs [MERT]
