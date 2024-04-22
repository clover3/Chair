import os

import cv2
import numpy as np

import cv2
import numpy as np
import cv2
import numpy as np

def parse_points(image_with_dot):
    # Define the red color range in RGB space
    lower_red = np.array([0, 0, 245])
    upper_red = np.array([10, 10, 256])
    # Create a mask for red color
    mask = cv2.inRange(image_with_dot, lower_red, upper_red)

    # Find non-zero pixels in the masked image
    points = cv2.findNonZero(mask)
    print(points)

    if points is not None and len(points) == 4:
        # Reshape the points array to have shape (4, 2)
        points = points.reshape(-1, 2)

        # Convert points to a list of tuples
        points = [tuple(point) for point in points]

        return points
    else:
        print("Error: Exactly four red pixels are required in the image.")
        return None


def align_images(image1, image1_with_dot, image2, image2_with_dot):
    # Parse points from image1_with_dot and image2_with_dot
    ref_points = parse_points(image1_with_dot)
    img_points = parse_points(image2_with_dot)

    if len(ref_points) == 4 and len(img_points) == 4:
        # Convert points to numpy arrays
        ref_points = np.array(ref_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # Find homography
        h, _ = cv2.findHomography(img_points, ref_points, cv2.RANSAC)

        # Use homography to align image
        height, width, _ = image1.shape
        img_aligned = cv2.warpPerspective(image2, h, (width, height))

        return img_aligned
    else:
        print("Insufficient or excessive points detected. Please ensure exactly four red dots are present in each image.")
        return None


def main():
    dir_path = r"C:\Users\leste\Pictures\construction"
    file_name_list = ["20240410_133611.jpg", "20240410_133611_mark.bmp",
                      "20240401_132819.jpg", "20240401_132819_mark.bmp"]

    file_path_list = [os.path.join(dir_path, name) for name in file_name_list]

    img1_path, img1_mark_path, img2_path, img2_mark_path = file_path_list
    image1 = cv2.imread(img1_path)
    image1_with_dot = cv2.imread(img1_mark_path)

    image2 = cv2.imread(img2_path)
    image2_with_dot = cv2.imread(img2_mark_path)

    # Align images
    aligned_image = align_images(image1, image1_with_dot, image2, image2_with_dot)

    if aligned_image is not None:
        # Save aligned image
        cv2.imwrite("aligned_image.jpg", aligned_image)
        print("Aligned image saved as 'aligned_image.jpg'")



if __name__ == "__main__":
    main()
# Read images
