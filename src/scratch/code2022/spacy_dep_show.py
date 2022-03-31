import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
sent1 = "Where does real insulin come from?"
sent2 = "Where does real phrase origin phrase come from?"
sent1 = "There is a book on the table."
sent2 = "We conclude that in women with preeclampsia, prolonged dietary supplementation with l-arginine significantly decreased blood pressure through increased endothelial synthesis and/or bioavailability of NO."

doc1 = nlp(sent1)
doc2 = nlp(sent2)
# Since this is an interactive Jupyter environment, we can use displacy.render here
items = [doc1, doc2]
displacy.serve(items, style='dep')