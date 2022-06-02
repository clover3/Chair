import os

index_name_list = [ "ClueWeb12-Disk1_00.idx",
    "ClueWeb12-Disk1_01.idx",
    "ClueWeb12-Disk1_02.idx",
    "ClueWeb12-Disk1_03.idx",
    "ClueWeb12-Disk1_04.idx",
    "ClueWeb12-Disk2_05.idx",
    "ClueWeb12-Disk2_06.idx",
    "ClueWeb12-Disk2_07.idx",
    "ClueWeb12-Disk2_08.idx",
    "ClueWeb12-Disk2_09.idx",
    "ClueWeb12-Disk3.idx",
    "ClueWeb12-Disk4_15.idx",
    "ClueWeb12-Disk4_16.idx",
    "ClueWeb12-Disk4_17.idx",
    "ClueWeb12-Disk4_18.idx",
    "ClueWeb12-Disk4_19.idx"]

index_dir = '/mnt/lustre/godzilla/harding/ClueWeb12/idx'


def get_first_disk():
    return os.path.join(index_dir, index_name_list[0])

