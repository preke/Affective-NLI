# Affective-NLI

## Data Construction


### FriendsPersona

1. `data/Friends_Persona.py`: construct unlabeled tsv data in each single dimension
2. `label_friends_persona_with_emoberta.py`: labeling utterance with basic emotion labels
3. `Friends_Persona_NLI_construction.py`: construct NLI data with different dialog flow lengths


### CPED


## Affective-NLI

- `src/affective_nli_roberta.py`: Run with RoBERTa
- `src/affective_nli_T5.py`: Run with T5
- `src/affective_nli_llama.py`: Run with llama2-7b 



Above three scripts share similar structures; dataset, random seeds, dialog length, hyperparameters can be modified in the main function (`if __name__ == '__main__':`).

