# Liar, Liar, LLM on Fire: Investigating Deception in Turkish Text Generation


## Repository Structure

- `dataset/` : Contains the curated TQuADFake dataset.  
- `code/` : Scripts for data generation, feature extraction, and experiments.  
- `evaluation/` : Evaluation for llm-eval in our study.  
- `figures/` : Figures and visualisations included in the paper.

## TQuADFake Dataset
We utilise the [TQuAD dataset](https://github.com/TQuad/turkish-nlp-qa-dataset), a Question Answer (QA) dataset in Turkish focusing on Islamic Science History. 

The TQuADFake dataset contains the following fields for each QA pair from TQuAD:  
- `question` : The original question from the TQuAD dataset.  
- `answer` : The corresponding answer.  
- `correct_statements` : A set of 5 sentences that correctly reflect the answer.  
- `incorrect_statements` : A set of 5 sentences that contradict or distort the answer.  

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/bolucunecva/turkish_liar.git
