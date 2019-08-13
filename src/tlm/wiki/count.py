from tf_util.count import count_instance

if __name__ == "__main__":
    working_path = "/mnt/nfs/work3/youngwookim/data/tlm_simple"

    cnt = count_instance("/mnt/nfs/work3/youngwookim/data/tlm_simple/tf")

    print(cnt)
    print(int(cnt/1000000) , " M")