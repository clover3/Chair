import json
from typing import List

from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.table_mem_check import load_table_as_indices


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse the JSON line
                json_data = json.loads(line)

                # Extract data_id, q_term_id, and d_term_ids
                data_id = json_data[0]
                q_term_id = json_data[1]
                d_term_ids = json_data[2]

                # Assuming the structure is [data_id, q_term_id, d_term_ids]
                if isinstance(data_id, int) and isinstance(q_term_id, int) and isinstance(d_term_ids, list):
                    e = {
                        'data_id': data_id,
                        'q_term_id': q_term_id,
                        'd_term_ids': d_term_ids
                    }
                    yield e
                else:
                    print(f"Invalid format in line: {line}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line} - {e}")


def load_table_per_worker(conf, worker_no):
    term_per_worker = 1000
    voca: List[str] = read_lines(conf.voca_path)
    voca_d = {t: idx for idx, t in enumerate(voca)}
    raw_table_scores_dir = conf.raw_table_scores
    st = term_per_worker * worker_no
    ed = st + term_per_worker
    c_log.info("Loading Tables")
    table_d: dict[int, dict[int, float]] = {}
    for q_term_i in range(st, ed):
        log_path = path_join(raw_table_scores_dir, f"{q_term_i}.txt")
        entry = load_table_as_indices(log_path, voca_d)
        table_d[q_term_i] = entry
    return table_d


def table_lookup(todo_path, save_path, table_d):
    top_k = 3
    c_log.info("Reading todo: %s", todo_path)
    parsed_data = read_json_file(todo_path)
    with open(save_path, "w") as out_f:
        # Output or process the parsed data
        for item in parsed_data:
            q_entry_d: dict[int, float] = table_d[item['q_term_id']]
            cands: list[tuple[int, float]] = []
            for d_term_id in item['d_term_ids']:
                try:
                    cands.append((d_term_id, q_entry_d[d_term_id]))
                except KeyError:
                    pass

            cands.sort(key=lambda x: x[1], reverse=True)
            out_e = {
                'data_id': item['data_id'],
                'q_term_id': item['q_term_id'],
                'cands': cands[:top_k]
            }
            s = json.dumps(out_e)
            out_f.write(s + "\n")
