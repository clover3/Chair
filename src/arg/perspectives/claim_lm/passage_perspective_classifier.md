


# Classifier
 * Input: perspective and passage

 
# There are three types of passages
 1. Good passage ( LM score > 0)
 2. Not good passage ( Top ranked which are not > 0)
 3. Random passage
 
 
# Output:
 * perspective + good passage : s > 1
 * perspective + random passage : s < -1
 * (perspective + good passage) > (perspective + not good passage)
 
   
   
 
# Data pipeline
 1. save_a_relevant.py
 2 datagen_passage_pers_classifier.py
   * TFrecord 