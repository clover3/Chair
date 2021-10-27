import json
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.resources import load_step1_claims
from arg.perspectives.load import load_perspectrum_golds, PerspectiveCluster
from clueweb.sydney_path import index_list
from cpath import at_output_dir


def load_pid_inv_index():
    out_d = {}
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()
    for cid, pc_list in gold_d.items():
        for pc in pc_list:
            min_pid = min(pc.perspective_ids)
            for pid in pc.perspective_ids:
                out_d[pid] = min_pid
    return out_d


def main():
    pid_inv_index = load_pid_inv_index()
    print(len(pid_inv_index))
    j_obj = load_step1_claims()
    old_query_list = json.load(open(at_output_dir("perspective_query", "pc_query_for_evidence.json"), "r"))

    old_query_d = {}
    for q in old_query_list["queries"]:
        old_query_d[q['number']] = q['text']
    print(len(old_query_d))

    new_queries = []
    for j in j_obj:
        try:
            cid = j['cid']
            ca_cid = str(j['ca_cid'])
            min_pid = pid_inv_index[j['perspective']['pid']]
            old_qid = "{}_{}".format(cid, min_pid)
            query_content = old_query_d[old_qid]
            query = {
                'number': ca_cid,
                'text': query_content
            }
            new_queries.append(query)
        except KeyError:
            print(old_qid, 'not found')

    st = 0
    step = 100

    while st < len(new_queries):
        target_queries = new_queries[st:st+step]
        st += step
        idx = int(st / step)
        out_path = at_output_dir("ca_building", "ca_build_queries.{}.json".format(idx))
        data = {
            "queries": target_queries,
            "index": index_list,
            "requested": 1000,
        }
        fout = open(out_path, "w")
        fout.write(json.dumps(data, indent=True))
        fout.close()


if __name__ == "__main__":
    main()