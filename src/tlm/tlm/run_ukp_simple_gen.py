import os

from data_generator.job_runner import sydney_working_dir
from list_lib import lmap
from tlm.data_gen.gen_from_txt import GenFromTxt


def work():
    root_path = "/mnt/nfs/work3/youngwookim/data/stance_small_docs_sents"
    path_list = [os.path.join(root_path, "abortion", "8.txt")
                , os.path.join(root_path, "cloning", "6.txt")
                , os.path.join(root_path, "death_penalty", "4.txt")
                , os.path.join(root_path, "marijuana_legalization", "4.txt")
                , os.path.join(root_path, "minimum_wage", "7.txt")
                , os.path.join(root_path, "nuclear_energy", "26.txt")
                , os.path.join(root_path, "school_uniforms", "4.txt")
                ]
    gen = GenFromTxt()
    docs = lmap(gen.load_doc, path_list)
    insts =gen.create_instances_from_documents(docs)
    out_path = os.path.join(sydney_working_dir, "ukp_sample", "1")
    gen.write_instance_to_example_files(insts, [out_path])


if __name__ == "__main__":
    work()