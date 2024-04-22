import numpy as np
import cv2
from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join


def align(img_path_list, save_path_list, anchor_list):
    assert len(img_path_list) == len(anchor_list)

    key_anchor = anchor_list[-1]["points"]
    key_anchor = np.array(key_anchor, dtype=np.float32)
    key_anchor = np.expand_dims(key_anchor, axis=1)
    print(key_anchor)

    key_img_path = img_path_list[-1]
    key_image = cv2.imread(key_img_path)

    for i in range(len(img_path_list)):
        image2 = cv2.imread(img_path_list[i])
        anchor = anchor_list[i]["points"]

        assert anchor_list[i]["name"] in img_path_list[i]
        anchor = np.array(anchor, dtype=np.float32)
        anchor = np.expand_dims(anchor, axis=1)

        print(anchor)

        # Find homography
        h, mask = cv2.findHomography(anchor, key_anchor, cv2.RANSAC)
        print(mask)

        # Use homography to align image
        height, width, _ = key_image.shape
        img_aligned = cv2.warpPerspective(image2, h, (width, height))
        cv2.imwrite(save_path_list[i], img_aligned)


def align_process(common_dir, src_dir_name, save_dir_name, file_list_path, anchor_name):
    name_list = [line.strip() for line in open(file_list_path, "r")]
    img_path_list = []
    save_path_list = []
    for name in name_list:
        name += ".jpg"
        img_path = path_join(common_dir, src_dir_name, name)
        img_path_list.append(img_path)
        save_path = path_join(common_dir, save_dir_name, name)
        save_path_list.append(save_path)
    anchor_path = path_join(common_dir, anchor_name)
    anchor_list = load_jsonl(anchor_path)
    align(img_path_list, save_path_list, anchor_list)



def main():
    common_dir = r"C:\Users\leste\Pictures\construction"
    file_list_path = path_join(common_dir, "man2.txt")
    src_dir_name = "man2"
    save_dir_name = "man2_align"
    anchor_name = "man2_anchors.txt"

    align_process(common_dir, src_dir_name, save_dir_name, file_list_path, anchor_name)



if __name__ == "__main__":
    main()