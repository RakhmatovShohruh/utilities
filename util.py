import argparse
import json
import os
import random
import shutil
from collections import Counter

import cv2


def get_sub_imgs(path, target_path):
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    os.makedirs(target_path, exist_ok=True)
    for dirs, _, files in os.walk(path):
        if dirs != path:
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    source = os.path.join(dirs, filename)

                    target = os.path.join(target_path, filename)
                    base, ext = os.path.splitext(target)
                    counter = 1
                    while os.path.exists(target):
                        target = f"{base}_{counter}{ext}"
                        counter += 1

                    shutil.move(source, target)


# Usage
# get_sub_imgs('data', 'data2')


def verify_name_pairs(image_path, label_path):
    # List directory contents
    images = os.listdir(image_path)
    labels = os.listdir(label_path)

    # Extract filenames without extensions and filter based on desired extensions
    image_names = [os.path.splitext(im)[0] for im in images if im.endswith(('.jpg', '.png', '.jpeg'))]
    label_names = [os.path.splitext(lab)[0] for lab in labels if lab.endswith('.json')]

    # Find unmatched image and label names
    diff_img_label = list((Counter(image_names) - Counter(label_names)).elements())
    diff_label_img = list((Counter(label_names) - Counter(image_names)).elements())

    # Report any mismatches
    if diff_img_label or diff_label_img or len(images) != len(labels):
        print("error comes from images folder:", diff_img_label)
        print("error comes from label folder:", diff_label_img)
        print('Please check folders.')
    else:
        print('OK...')


# Usage
# verify_name_pairs('split_test', 'split_test')


def visualize_landmarks(image_name, pts_name, output_name=None):
    with open(pts_name, 'r') as file:
        content = file.readlines()

    start_idx = next(i for i, line in enumerate(content) if "{" in line) + 1
    end_idx = next(i for i, line in enumerate(content) if "}" in line)
    landmarks_data = content[start_idx:end_idx]

    landmarks = [tuple(map(float, line.split())) for line in landmarks_data]

    img = cv2.imread(image_name)

    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)  # Drawing a red circle for each point
        cv2.putText(img, str(idx), (int(x) + 2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imshow("Face Landmarks", img)

    if output_name:
        cv2.imwrite(output_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage
# visualize_landmarks(image_name, pts_name, '1.jpg')


def split_dataset(main_folder, train_folder, test_folder, test_ratio=0.2):
    image_extensions = ['.jpg']
    label_extensions = ['.json']
    # Ensure the train and test folders exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all image files from the main folder
    image_files = [f for f in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, f)) and any(
        f.lower().endswith(ext) for ext in image_extensions)]

    # Shuffle and split the files
    random.shuffle(image_files)
    split_index = int(len(image_files) * (1 - test_ratio))
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Move files function (nested within the main function)
    def move_files(files, source_folder, destination_folder):
        for filename in files:
            # Move the image file
            shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))

            # Move the corresponding label file
            base_name, _ = os.path.splitext(filename)
            for ext in label_extensions:
                label_file = f"{base_name}{ext}"
                label_path = os.path.join(source_folder, label_file)
                if os.path.exists(label_path):
                    shutil.move(label_path, os.path.join(destination_folder, label_file))

    # Call the move files function for train and test data
    move_files(train_files, main_folder, train_folder)
    move_files(test_files, main_folder, test_folder)


# Usage
# main_dir = "split"
# train_dir = "split_train"
# test_dir = "split_test"
#
# split_dataset(main_dir, train_dir, test_dir)


def collect_files(src1, src2, destination):
    os.makedirs(destination, exist_ok=True)

    def copy_from_source(src):
        for filename in os.listdir(src):
            file_path = os.path.join(src, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(destination, filename))

    copy_from_source(src1)
    copy_from_source(src2)


# Usage
# image_folder = "temp/image"
# label_folder = "temp/anno"
# destination_folder = "temp2"
#
# collect_files(image_folder, label_folder, destination_folder)


def json2pts(root, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    def json_to_pts(json_data):
        keypoints = json_data["ObjectInfo"]["KeyPoints"]["Points"]

        if len(keypoints) % 2 != 0:
            raise ValueError("Unexpected number of landmark points. They should be in pairs.")

        n_points = len(keypoints) // 2

        # Convert JSON data to PTS format
        pts_data = []
        pts_data.append("version: 1")
        pts_data.append(f"n_points:  {n_points}")
        pts_data.append("{")
        for i in range(0, len(keypoints), 2):
            x = keypoints[i]
            y = keypoints[i + 1]
            pts_data.append(f"{x} {y}")
        pts_data.append("}")

        return "\n".join(pts_data)

    # Walk through the root directory
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = os.path.join(dirpath, filename)

                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)

                pts_content = json_to_pts(data)
                pts_path = os.path.join(dest_dir, filename.replace('.json', '.pts'))

                with open(pts_path, 'w') as pts_file:
                    pts_file.write(pts_content)


# Usage
# root_dir = "./temp/anno"
# dest_dir = "./temp/pts"
# json2pts(root_dir, dest_dir)

if __name__ == "__main__":
    a = argparse.ArgumentParser(description='Data Processing for Landmark points')
    a.add_argument("--path1", help="path to the data")
    a.add_argument("--path2", help="path to the data")
    a.add_argument("--path3", help="path to the data")
    a.add_argument("--image-path", help="image path")
    a.add_argument("--label-path", help="label path")
    a.add_argument("--mode", help="json2pts", required=True)

    args = a.parse_args()
    print(args)

    if args.mode == "json2pts":
        print("===== Convert Json To pts =====")
        json2pts(args.path1, args.path2)

    elif args.mode == "collect_files":
        print("===== Collecting Files =====")
        collect_files(args.path1, args.path2, args.path3)
        """
        copy (not move) all the files from the two source directories to the destination directory. 
        """
    elif args.mode == "split":
        print("===== Split Dataset =====")
        split_dataset(args.path1, args.path2, args.path3)
        """
        split data into train and test folder (Train-80%, Test-20%)
        """
    elif args.mode == "visualize":
        print("===== Visualizing Facial Keypoints =====")
        visualize_landmarks(args.image_path, args.label_path)
        """
        Visualize facial landmarks on an image and optionally save the output.
        """
    elif args.mode == "verify_names":
        print("===== Visualizing Facial Keypoints =====")
        visualize_landmarks(args.image_path, args.label_path)
        """
        Verify name of files (compare image name with label name)
        """
    elif args.mode == "get_sub_image":
        print("===== collect sub folders' images=====")
        get_sub_imgs(args.path1, args.path2)
        """
        Collecting files from sub folders
        """

    else:
        print("mode error : You can only use video information inquiry and video frame extraction.")

ChatGPT