
def main():
    raw_data = """
Exact match/neutral/acc:1.01
Exact match/neutral/f1:0.01
Exact match/contradiction/acc:1.01
Exact match/contradiction/f1:0.00
word2vec/neutral/acc:0.98
word2vec/neutral/f1:0.63
word2vec/contradiction/acc:1.01
word2vec/contradiction/f1:0.00
Co-attention/neutral/acc:1.00
Co-attention/neutral/f1:0.97
LIME/neutral/acc:0.55
LIME/neutral/f1:-0.05
LIME/contradiction/acc:0.10
LIME/contradiction/f1:-0.40
Occlusion/neutral/acc:0.95
Occlusion/neutral/f1:-5.00
Occlusion/contradiction/acc:0.95
Occlusion/contradiction/f1:-5.00
SE-NLI/neutral/acc:4.95
SE-NLI/neutral/f1:-2.05
SE-NLI/contradiction/acc:4.90
SE-NLI/contradiction/f1:0.25
Token-entail/neutral/acc:0.99
Token-entail/neutral/f1:0.04
Token-entail/contradiction/acc:0.95
Token-entail/contradiction/f1:0.28
PAT/neutral/acc:0.67
PAT/neutral/f1:0.15
PAT/contradiction/acc:0.45
PAT/contradiction/f1:0.18
"""
    d = {}
    method_list = []
    for line in raw_data.split("\n"):
        if line.strip():
            head, score = line.split(":")
            method, tag, metric = head.split("/")
            d[method, tag, metric] = score
            if method not in method_list:
                method_list.append(method)


    print("{0:12s}|{1:5s}|{2:5s}|{3:5s}|{4:4s}".format("", "f1", "acc", "f1", "acc"))

    for method in method_list:
        row = [method]
        for tag in ["neutral", "contradiction"]:
            for metric in ["f1", "acc"]:
                try:
                    s = d[method, tag, metric]
                except:
                    s = "-"
                row.append(s)
        print("{0:12s}|{1:5s}|{2:5s}|{3:5s}|{4:4s}".format(*row))



if __name__ == "__main__":
    main()