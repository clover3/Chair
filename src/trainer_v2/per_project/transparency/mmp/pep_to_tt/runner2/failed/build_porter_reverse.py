from collections import defaultdict

from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.misc_common import save_tsv

class NonExactError(Exception):
    pass


def main():
    term_path = path_join(output_path, "mmp", "lucene_standard_terms.txt")
    mapping_save_path = path_join(output_path, "mmp", "porter_to_krovetz.txt")
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    k_analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
    p_analyzer = Analyzer(get_lucene_analyzer(stemmer='porter'))

    f_in = open(term_path, "r")


    porter_mapping = {}
    porter_mapping_per_stem = defaultdict(list)
    root_p_to_k = {}
    for line in f_in:
        term = line.strip()
        try:
            k_tokens = k_analyzer.analyze(term)
            p_tokens = p_analyzer.analyze(term)

            if len(k_tokens) == len(p_tokens):
                for kt, pt in zip(k_tokens, p_tokens):

                    if pt != term:
                        porter_mapping[term] = kt
                        porter_mapping_per_stem[kt].append(term)

                    # if krovetz_stemmed == term:
                    #     print(f"In Porter {term} -> {porter_stemmed} but in Krovetz still {krovetz_stemmed}")

                        # if pt in root_p_to_k:
                        #     if root_p_to_k[pt] != kt:
                        #         print(f"Before\t{term}/{pt}/{root_p_to_k[pt]}")
                        #         print(f"Now\t{term}/{pt}/{kt}")
                        if kt != pt:
                            root_p_to_k[pt] = kt
        except NonExactError as e:
            print(e)


    save_tsv(root_p_to_k.items(), mapping_save_path)


if __name__ == "__main__":
    main()