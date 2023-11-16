from scipy.spatial.distance import cosine

from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.basic_embedding.bert_encoder_wrap import \
    BERTEncoderWrap


def dev():
    encoder = BERTEncoderWrap()
    # Get the embedding for '1997'
    word_embedding_1997 = encoder.batch_encode('1997')

    return
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
