from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs
from arg.qck.instance_generator.qknc_datagen import QKInstanceGenerator

# Make payload without any annotation
from exec_lib import run_func_with_config


def main(config):
    def is_correct(dummy_query, dummy_passage):
        return 0

    start_generate_jobs(QKInstanceGenerator(is_correct),
                        config['split'],
                        config['qk_candidate_name'],
                        config['name_prefix'])


if __name__ == "__main__":
    run_func_with_config(main)