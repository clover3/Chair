import numpy as np
import cv2
from cpath import output_path
from misc_lib import path_join


def adjust_color_histogram(images):
    # Convert images to the LAB color space
    lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]

    # Compute the average histogram of the LAB images
    hist_size = [256, 256, 256]
    hist_range = [0, 256, 0, 256, 0, 256]
    avg_hist = np.zeros((hist_size[0], hist_size[1], hist_size[2]), dtype=np.float32)
    for lab_img in lab_images:
        hist = cv2.calcHist([lab_img], [0, 1, 2], None, hist_size, hist_range)
        avg_hist += hist
    avg_hist /= len(lab_images)

    # Match the histogram of each image to the average histogram
    matched_images = []
    for lab_img in lab_images:
        matched_lab_img = np.copy(lab_img)
        for i in range(3):
            channel = lab_img[:, :, i]
            matched_channel = cv2.LUT(channel, cv2.normalize(avg_hist[:, :, i].flatten(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            matched_lab_img[:, :, i] = matched_channel
        matched_img = cv2.cvtColor(matched_lab_img, cv2.COLOR_LAB2BGR)
        matched_images.append(matched_img)

    return matched_images

def adjust_color(common_dir, src_dir_name, save_dir_name, file_list_path):
    name_list = [line.strip() for line in open(file_list_path, "r")]
    images = []
    for name in name_list:
        name += ".jpg"
        img_path = path_join(common_dir, src_dir_name, name)
        image = cv2.imread(img_path)
        images.append(image)

    adjusted_images = adjust_color_histogram(images)

    for image, name in zip(adjusted_images, name_list):
        name += ".jpg"

        save_path = path_join(common_dir, save_dir_name, name)
        cv2.imwrite(save_path, image)


def main():
    common_dir = r"C:\Users\leste\Pictures\construction"
    file_list_path = path_join(common_dir, "nozoom.txt")
    src_dir_name = "sel1_align_crop"
    save_dir_name = "sel1_align_crop_color_adjusted"
    adjust_color(common_dir, src_dir_name, save_dir_name, file_list_path, )



if __name__ == "__main__":
    main()