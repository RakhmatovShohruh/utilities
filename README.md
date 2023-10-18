**Table of Contents:**
* **Introduction**
* **Functions**
* * get_sub_imgs
* * verify_name_pairs
* * visualize_landmarks
* * split_dataset
* * collect_files
* * json2pts
* **Usage**


**Introduction:**
This Python script provides a set of functions to process and manage data for facial landmark recognition tasks. 

**Functions**

1. **get_sub_imgs:**
This function collects image files with specified extensions from subfolders and moves them to a target directory while renaming any conflicting filenames.

2. **verify_name_pairs:**
Verifies the name pairs between image and label files in two directories and checks for any mismatches. It reports any discrepancies in file names.

3. **visualize_landmarks:**
Visualizes facial landmarks on an image and optionally saves the output. This function is useful for visual inspection of landmark data.

4. **split_dataset:**
Splits a dataset into training and testing folders, with a customizable test ratio. It moves both image and label files to their respective folders.

5. **collect_files:**
Collects (copies, not moves) all files from two source directories and places them in a destination directory.

6. **json2pts:**
Converts JSON label files to PTS format, making them suitable for use in facial landmark recognition tasks.

**Here are some examples of how to use the script in different modes:**
* Converting JSON label files to PTS format:                                                                   
$ python script.py --mode json2pts --path1 ./data --path2 ./output

