import tensorflow as tf

def file_sampler(filepath, file_byte_budget=1e6):
    with tf.gfile.GFile(filepath, mode="r") as source_file:
        file_byte_budget_ = file_byte_budget
        counter = 0
        countermax = int(source_file.size() / file_byte_budget_ / 2)
        for line in source_file:
            if counter < countermax:
                counter += 1
            else:
                if file_byte_budget_ <= 0:
                    break
                line = line.strip()
                file_byte_budget_ -= len(line)
                counter = 0
                yield line



def init_shared_voca():
    # TODO load Stance Data

    # TODO get pointer to wiki

    wiki_path = NotImplemented

