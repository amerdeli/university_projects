# Assignment 3 — Generative Pre-trained Transformer (GPT)

## Overview
This assignment implements a **character-level GPT model** using only PyTorch primitives.  
The goal is to understand and build the core components of a Transformer from scratch.

## Tasks Completed
- Analyzed the dataset and dataloader behavior.
- Implemented:
  - Causal multi-head self-attention
  - MLP block
  - GPT forward pass
  - Sampling with temperature
- Explained Q/K/V matrices, masking, and attention mechanics.
- Trained the model and plotted training/validation loss curves.
- Generated text samples for multiple temperature values.
- Tested different prompts and analyzed model behavior.
- Scaled the model (depth, embedding size, heads, block size) and compared:
  - Parameter counts  
  - Training dynamics  
  - Sample quality  
  - Computational requirements  

## Files
- `gpt_assignment.ipynb` — full implementation of the GPT model, training, sampling, and experiments
- - `gpt_assignment_report.pdf` — full implementation of the GPT model, training, sampling, and experiments

