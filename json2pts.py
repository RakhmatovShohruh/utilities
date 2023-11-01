import os
import json

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

