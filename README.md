# progress-measures-paper

`transformers.py` contains the code to train the model. `Grokking_Analysis.ipynb` contains the code to load the saved checkpoints for the mainline run, calculate the progress metrics on it, and plots the figures. `Non_Modular_Addition_Grokking_Tasks.ipynb` contains training code for the non-modular addition experiments.

# To Do:
- For charcter level tokenization with varying lenghts (e.g. 12+31=43EOSPADPADPADPAD) do an accuracy table for all cases (e.g. |operator 1|= |operator 2| = 1; |operator 1| = 1,  |operator 2| = 2) etc) on train and test, maybe do different seeds and plot a distribution for each category
- Try character level tokenization with 0s to fix the varying length (e.g. 012+031=043EOSPAD), do another accuracy table
- Add a seed() to fix the train and test set [DONE]
- Vary positional encodings (current implementations learns the positional encodings)
- Change folder structure and save the config and dataset for different runs [Aylin]
- Add the conda environment file [Mert]
- Add a saving scheduling function (save a lot first 50 epochs and then save less)
- Add something to upsample lower-digit case (and potentially 3 digit in our case) [Aylin]
- Train successful model. [BOTH]

# To Read

Positional encoding related stuff:

- https://arxiv.org/pdf/2405.17399v1 (Transformers Can Do Arithmetic with the
Right Embeddings)
- https://arxiv.org/pdf/2305.19466 The Impact of Positional Encoding on Length
Generalization in Transformers (NoPE paper)

Length Generalization and RASP Papers (should be read in this order imo):
- https://arxiv.org/pdf/2106.06981 (RASP Thinking Like Transformers)
- https://arxiv.org/abs/2310.16028 What Algorithms can Transformers Learn? A Study in Length Generalization
- https://openreview.net/forum?id=FmhPg4UJ9K#discussion (Counting Like Transformers: Compiling Temporal Counting Logic Into Softmax Transformers)

Additional, not super important:
- https://arxiv.org/pdf/2406.06467 (How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad)

Interpretability: 

- https://arxiv.org/pdf/2406.16778 (Finding Transformer Circuits with Edge Pruning)


