import numpy as np
import cv2


from cpath import output_path
from misc_lib import path_join




def crop_save(common_dir, src_dir_name, save_dir_name, file_list_path, border):
    (left, top), (right, bottom) = border
    name_list = [line.strip() for line in open(file_list_path, "r")]
    img_path_list = []
    save_path_list = []
    for name in name_list:
        name += ".jpg"
        img_path = path_join(common_dir, src_dir_name, name)
        save_path = path_join(common_dir, save_dir_name, name)

        image = cv2.imread(img_path)
        image_crop = image[top:bottom, left:right]
        cv2.imwrite(save_path, image_crop)


def main():
    common_dir = r"C:\Users\leste\Pictures\construction"
    file_list_path = path_join(common_dir, "nozoom.txt")
    src_dir_name = "sel1_align"
    save_dir_name = "sel1_align_crop"
    border = [565, 669], [3527, 2846]
    crop_save(common_dir, src_dir_name, save_dir_name, file_list_path, border)



if __name__ == "__main__":
    main()