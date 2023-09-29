import json
import os
import glob
import tqdm
import cv2
import numpy as np


classes_dict = {'call': 0, 'dislike': 0, 'fist': 0, 'four': 0, 'like': 0, 'mute': 0, 'ok': 0, 'one': 0, 'palm': 0, 'peace': 0, 'peace_inverted': 0, 'rock': 0, 'stop': 0, 'stop_inverted': 0, 'three': 0, 'three2': 0, 'two_up': 0, 'two_up_inverted': 0, 'no_gesture': 0}


def crop_data(read_path, write_path):
    json_file_path = glob.glob(os.path.join(read_path, "**/*.json"))
    assert len(json_file_path) > 0, "No json files are found in your data files."

    if not os.path.exists(write_path):
        os.mkdir(write_path)
        os.mkdir(os.path.join(write_path, "train"))

    for json_path in json_file_path:
        file_path = os.path.split(json_path)[0]
        f = open(json_path)
        data = json.load(f)

        train_save_path = os.path.join(write_path, "train", os.path.split(file_path)[-1])
        if not os.path.exists(train_save_path):
            os.mkdir(train_save_path)

        # --------------------------
        # Creating Train Data
        # --------------------------
        annotations = {}
        for file_name in tqdm.tqdm(list(data.keys())):
            bboxes = data[file_name]["bboxes"]
            labels = data[file_name]["labels"]
            landmarks = data[file_name]["landmarks"]

            for bbox, label, landmark in zip(bboxes, labels, landmarks):
                if label in classes_dict and len(landmark) > 0:
                    img = cv2.imread(os.path.join(file_path, file_name + ".jpg"), cv2.IMREAD_COLOR)

                    new_file_name = label + "_" + str(classes_dict[label])
                    classes_dict[label] += 1

                    height, width, _ = img.shape

                    x1, y1, w, h = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)
                    x3, y3 = x1 + w, y1 + h # bottom right corner

                    landmark = np.array(landmark)
                    landmark[:, 0] = landmark[:, 0] * width
                    landmark[:, 1] = landmark[:, 1] * height
                    
                    hand = img[y1:y3, x1:x3]
                    landmark[:, 0] = landmark[:, 0] - x1
                    landmark[:, 1] = landmark[:, 1] - y1
                    width, height = w, h

                    landmark[:, 0] = landmark[:, 0] / width
                    landmark[:, 1] = landmark[:, 1] / height

                    cv2.imwrite(os.path.join(train_save_path, new_file_name + ".jpg"), hand)

                    annotations[new_file_name] = {}
                    annotations[new_file_name]["bboxes"] = [[0, 0, 0, 0]]
                    annotations[new_file_name]["labels"] = [label]
                    annotations[new_file_name]["landmarks"] = [landmark.tolist()]

        with open(os.path.join(train_save_path, os.path.split(json_path)[1]), "w") as outfile:
            json.dump(annotations, outfile, ensure_ascii=False, indent=4)

        f.close()


if __name__ == "__main__":
    crop_data("data/hagrid/train", "data/hagrid_crop")