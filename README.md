# Lexical Substitution
This is a homework submission for COMSW4705 Natural Language Processing.

This program accesses WordNet with NLTK, which can be installed by e.g. `pip install nltk`, gensim by e.g. `pip install gensim`, DistilBERT from Huggingface with first `pip install transformers` and the model accessed by `transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')`. It used a several-GB [pre-trained word embeddings](https://drive.google.com/u/1/uc?id=0B7XkCwpI5KDYNlNUTTISS21pQmM&export=download) (link may break in the future).

The lexical substitution task is attempted with a simple best WordNet frequency from context, a simple Lesk algorithm, most similar synonym with Word2Vec embeddings, and a BERT masked language model. The predictions were written to each `.predict` file, which can be evaluated with the `score.pl` script (`perl score.pl smurf.predict gold.trial`).
