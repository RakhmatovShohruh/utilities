import os
import cv2
import glob
import shutil
import numpy as np
import xml.etree.ElementTree as ET

def batch_convert_xml_to_pts_robust(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    
    for xml_file in xml_files:
        try:
            pts_filename = os.path.basename(xml_file).replace('.xml', '.pts')
            pts_filepath = os.path.join(output_dir, pts_filename)
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            face_68_elem = root.find('.//face_68')
            if face_68_elem is not None and 'points' in face_68_elem.attrib:
                points_str = face_68_elem.attrib['points']
                points_list = points_str.split(';')
                points = [[float(coord) for coord in point.split(',')[:2]] for point in points_list if len(point.split(',')) >= 2]
                if len(points) != 68:
                    print(f"Warning: Expected 68 points, but found {len(points)} in {os.path.basename(xml_file)}")
            else:
                print(f"No facial landmark points found in {os.path.basename(xml_file)}")
                continue
            
            points = np.array(points)
            
            with open(pts_filepath, 'w') as f:
                f.write("version: 1\n")
                f.write(f"n_points:  {len(points)}\n")
                f.write("{\n")
                for point in points:
                    f.write(f"{point[0]} {point[1]}\n")
                f.write("}\n")
            
            print(f"Successfully converted: {os.path.basename(xml_file)} -> {pts_filename}")
        except Exception as e:
            print(f"Error processing {os.path.basename(xml_file)}: {str(e)}")
    
    print("Batch conversion completed.")


def count_facial_landmarks_pts(pts_file_path):
    try:
        with open(pts_file_path, 'r') as f:
            lines = f.readlines()
            n_points_line = next((line for line in lines if line.startswith("n_points")), None)
            if n_points_line:
                # Extract the number of points
                n_points = int(n_points_line.split(":")[1].strip())
                return n_points
            else:
                print(f"Error: 'n_points' not found in {os.path.basename(pts_file_path)}")
                return None
    except Exception as e:
        print(f"Error processing {os.path.basename(pts_file_path)}: {str(e)}")
        return None

def check_pts_files(directory):
    pts_files = glob.glob(os.path.join(directory, '*.pts'))
    
    for pts_file in pts_files:
        n_points = count_facial_landmarks_pts(pts_file)
        if n_points is not None and n_points != 68:
            print(f"{os.path.basename(pts_file)}: {n_points} points")
            