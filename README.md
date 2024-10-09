# progress-measures-paper

`transformers.py` contains the code to train the model. `Grokking_Analysis.ipynb` contains the code to load the saved checkpoints for the mainline run, calculate the progress metrics on it, and plots the figures. `Non_Modular_Addition_Grokking_Tasks.ipynb` contains training code for the non-modular addition experiments.

# To Do:
- Train a Linear Classifier for Carry Digits from 1) resid_mid and 2) resid_post [DONE]
  - Insights:
  - There is a carry feature in the learned representations. We can predict the carry with > 99% accuracy
  - Attention already extracts carry digits feature, because we can predict it from resid_mid
  - resid_mid and resid_post have different "basis", because carry digits basis vectors do not align
- Decode from resid_mid [DONE]:
    - Model completely fails, probably because resid_mid is not aligned with the unembedding space
- PCA of learned embeddings (algebraic structure visible), compare over epochs
- Add a functionality to train from checkpoint (side quest, would save us if training is interrupted somehow)
- Vary positional encodings (current implementations learns the positional encodings)

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
- https://aclanthology.org/2023.acl-long.893.pdf (Analyzing Transformers in Embedding Space)
- https://arxiv.org/abs/2402.11917 (A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task)


