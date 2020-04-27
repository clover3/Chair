from tlm.alt_emb.xml_query_to_json import xml_query_to_json


def main():
    xml_path = "/mnt/nfs/work3/youngwookim/code/Chair/data/CLEFeHealth2017IRtask/queries/queries2016.xml"
    json_path = "/mnt/nfs/work3/youngwookim/code/Chair/data/CLEFeHealth2017IRtask/queries/queries2016.json"
    xml_query_to_json(xml_path, json_path)


if __name__ == "__main__":
    main()
