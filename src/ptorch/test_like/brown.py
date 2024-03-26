import nltk

# Download the Brown Corpus
nltk.download('brown')
# Import the Brown Corpus
from nltk.corpus import brown

# Access the text of the Brown Corpus
brown_text = brown.sents()  # This gives you the raw text of the corpus

for t in brown_text:
    print(" ".join(t))