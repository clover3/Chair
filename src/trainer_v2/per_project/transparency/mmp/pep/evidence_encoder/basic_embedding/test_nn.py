import random

from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.basic_embedding.nn_helper import \
    get_embedding_nn_index


def main(dir_name):
    embeddings, index, words = get_embedding_nn_index(dir_name)
    random.seed(0)
    for target_word in random.choices(words[:1000], k=10):
        idx = words.index(target_word)
        query_vector = embeddings[idx]
        query_vector = query_vector.reshape(1, -1)  # Reshape for FAISS compatibility

        # Number of top similar vectors to retrieve
        n = 1
        k = 10
        # Performing the search
        # D, I = index.search(n=n, x=query_vector, k=k, distances=dist, labels=labels)  # D is the distance, I is the index of vectors
        D, I = index.search(query_vector, k)
        # Display the results
        # print("Top 10 similar vectors (inner product and index):")
        neighbor_terms = []
        for i in range(k):
            idx = I[0][i]
            word = words[idx]
            neighbor_terms.append(word)
            # print(f"{idx}: {word}, Inner product: {D[0][i]}")

        print("{}: {} ".format(target_word, neighbor_terms))


if __name__ == "__main__":
    dir_name = "bert_encoding_first"
    main("bert_encoding_pooled")
    main("bert_encoding_first")
    main("bert_encoding_max")