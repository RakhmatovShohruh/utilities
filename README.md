This README provides an overview of the Python script and its functions for data processing related to facial landmark points. The script includes various utilities to assist in managing and processing image and label data for facial landmark recognition tasks.

Table of Contents
Introduction
Prerequisites
Functions
1. get_sub_imgs
2. verify_name_pairs
3. visualize_landmarks
4. split_dataset
5. collect_files
6. json2pts
Usage
Command-line Arguments
Examples


Introduction
This Python script provides a set of functions to process and manage data for facial landmark recognition tasks. It includes utilities for tasks such as collecting image files from subfolders, verifying name pairs between image and label files, visualizing facial landmarks on images, splitting datasets into training and testing sets, and converting JSON label files to PTS format.

Prerequisites
Before using this script, ensure that you have the following prerequisites installed:

Python 3.x
OpenCV (cv2 Python library)
You can install OpenCV using pip:

bash
Copy code
pip install opencv-python
Functions
1. get_sub_imgs
This function collects image files with specified extensions from subfolders and moves them to a target directory while renaming any conflicting filenames.

2. verify_name_pairs
Verifies the name pairs between image and label files in two directories and checks for any mismatches. It reports any discrepancies in file names.

3. visualize_landmarks
Visualizes facial landmarks on an image and optionally saves the output. This function is useful for visual inspection of landmark data.

4. split_dataset
Splits a dataset into training and testing folders, with a customizable test ratio. It moves both image and label files to their respective folders.

5. collect_files
Collects (copies, not moves) all files from two source directories and places them in a destination directory.

6. json2pts
Converts JSON label files to PTS format, making them suitable for use in facial landmark recognition tasks.

Usage
To use the script, you can import and call the functions in your Python code. Additionally, you can run the script from the command line with various modes and command-line arguments.

Command-line Arguments
The script supports several command-line arguments for different modes of operation:

--path1: Path to the data (used in various functions).
--path2: Path to the data (used in various functions).
--path3: Path to the data (used in the collect_files function).
--image-path: Path to the image file (used in the visualize_landmarks function).
--label-path: Path to the label file (used in the visualize_landmarks function).
--mode: Mode of operation (e.g., json2pts, collect_files, split, visualize, verify_names, get_sub_image).
Examples
Here are some examples of how to use the script in different modes:

Converting JSON label files to PTS format:

bash
Copy code
python script.py --mode json2pts --path1 ./data --path2 ./output
Collecting files from two source directories and copying them to a destination directory:

bash
Copy code
python script.py --mode collect_files --path1 ./source_dir1 --path2 ./source_dir2 --path3 ./destination_dir
Splitting a dataset into training and testing sets:

bash
Copy code
python script.py --mode split --path1 ./main_data --path2 ./train_data --path3 ./test_data
Visualizing facial landmarks on an image:

bash
Copy code
python script.py --mode visualize --image-path ./image.jpg --label-path ./labels.json
Verifying name pairs between image and label files:

bash
Copy code
python script.py --mode verify_names --image-path ./image_folder --label-path ./label_folder
Collecting image files from subfolders:

bash
Copy code
python script.py --mode get_sub_image --path1 ./data --path2 ./output
Feel free to adapt these commands to your specific use case.

