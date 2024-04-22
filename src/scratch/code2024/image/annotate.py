import json
import shutil

import cv2
import numpy as np
from cpath import output_path
from misc_lib import path_join


def annotate(refFilename):
    # Global variables
    ref_points = []
    n_ref = 4
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(ref_points) < n_ref:
                ref_points.append((x, y))
                cv2.circle(ref_image, (x, y), 3, (0, 255, 0), -1)
            update_image_display()

    def update_image_display():
        cv2.imshow("Image Alignment", ref_image)

    # Read reference image
    print("Reading reference image : ", refFilename)
    ref_image = cv2.imread(refFilename)

    # Create a scrollable window for image alignment
    cv2.namedWindow("Image Alignment", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image Alignment", mouse_click)

    update_image_display()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(ref_points) == n_ref:
            break
        if key == ord("c"):
            ref_points = []
            ref_image = cv2.imread(refFilename)
            update_image_display()

    cv2.destroyAllWindows()
    return ref_points

def make_anchors():
    common_dir = r"C:\Users\leste\Pictures\construction"
    file_list_path = path_join(common_dir, "man2.txt")

    name_list = [line.strip() for line in open(file_list_path, "r")]
    log_path = path_join(common_dir, "man2_anchors.txt")
    log_f = open(log_path, "w")
    for name in name_list:
        name += ".jpg"
        img_path = path_join(common_dir, "man2", name)
        points = annotate(img_path)
        row_j = {"name": name, "points": points}
        log_f.write(json.dumps(row_j) + "\n")


def move_files():
    common_dir = r"C:\Users\leste\Pictures\construction"
    file_list_path = path_join(common_dir, "nozoom.txt")
    name_list = [line.strip() for line in open(file_list_path, "r")]
    for name in name_list:
        name += ".jpg"
        img_path = path_join(common_dir, "until0413", name)
        trg_path = path_join(common_dir, "sel1", name)
        shutil.move(img_path, trg_path)


if __name__ == '__main__':
    make_anchors()