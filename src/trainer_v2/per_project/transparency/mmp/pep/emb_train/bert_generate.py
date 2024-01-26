from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get the embedding of a word
def get_word_embedding(word):
    input_ids = tokenizer.encode(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)

    out_a, out_b = outputs
    return outputs[1]


def main():
    # Get the embedding for '1997'
    word_embedding_1997 = get_word_embedding('1997')

    # List of candidate words (can be expanded)
    # candidate_words = ['1998', 'technology', 'history', 'car', 'music', 'economy']
    candidate_words = [str(s) for s in range(1, 2023)]
    # candidate_words = []

    # Calculate similarity with each candidate word
    similarities = {}
    for word in candidate_words:
        word_embedding = get_word_embedding(word)
        similarity = 1 - cosine(word_embedding_1997, word_embedding)
        similarities[word] = similarity

    # Sort words by similarity
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted similar words
    for word, similarity in sorted_words[:100]:
        print(f"Word: {word}, Similarity: {similarity}")


if __name__ == "__main__":
    main()
