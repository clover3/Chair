

def main():
    # HINT
    def generate_type(q, d):
        pass

    def generate_type_b(q, q_a):
        pass

    q = "what is A"
    q_a = "what is"
    d = "document"
    t = generate_type_b(q, q_a)
    t, d_mod = generate_type(q, d)

    def relevance(q, d_mod):
        pass

    score_pred = relevance(q, d_mod)
    score_target = relevance(q, d)
    loss = score_pred - score_target
    return NotImplemented


if __name__ == "__main__":
    main()