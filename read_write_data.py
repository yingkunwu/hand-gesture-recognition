import json
import os
import glob
import shutil
import argparse


def read_write_data(read_path, write_path, num_of_data):
    # This function copy same data from original data file to custom data file
    # The goal is to create smaller dataset that can save spaces
    json_file_path = glob.glob(os.path.join(read_path, "**/*.json"))
    assert len(json_file_path) > 0, "No json files are found in your data files."

    if not os.path.exists(write_path):
        os.mkdir(write_path)
        os.mkdir(os.path.join(write_path, "train"))
        os.mkdir(os.path.join(write_path, "test"))

    num_of_train_data = int(num_of_data * 0.9)

    for json_path in json_file_path:
        file_path = os.path.split(json_path)[0]
        f = open(json_path)
        data = json.load(f)

        train_save_path = os.path.join(write_path, "train", os.path.split(file_path)[-1])
        if not os.path.exists(train_save_path):
            os.mkdir(train_save_path)

        test_save_path = os.path.join(write_path, "test", os.path.split(file_path)[-1])
        if not os.path.exists(test_save_path):
            os.mkdir(test_save_path)

        # --------------------------
        # Creating Train Data
        # --------------------------
        annotations = {}
        for file_name in list(data.keys())[:num_of_train_data]:
            bbox = data[file_name]["bboxes"]
            labels = data[file_name]["labels"]
            landmarks = data[file_name]["landmarks"]

            annotations[file_name] = {}
            annotations[file_name]["bboxes"] = bbox
            annotations[file_name]["labels"] = labels
            annotations[file_name]["landmarks"] = landmarks

            shutil.copyfile(os.path.join(file_path, file_name + ".jpg"), os.path.join(train_save_path, file_name + ".jpg"))

        with open(os.path.join(train_save_path, os.path.split(json_path)[1]), "w") as outfile:
            json.dump(annotations, outfile, ensure_ascii=False, indent=4)

        # --------------------------
        # Creating Test Data
        # --------------------------
        annotations = {}
        for file_name in list(data.keys())[num_of_train_data:num_of_data]:
            bbox = data[file_name]["bboxes"]
            labels = data[file_name]["labels"]
            landmarks = data[file_name]["landmarks"]

            annotations[file_name] = {}
            annotations[file_name]["bboxes"] = bbox
            annotations[file_name]["labels"] = labels
            annotations[file_name]["landmarks"] = landmarks

            shutil.copyfile(os.path.join(file_path, file_name + ".jpg"), os.path.join(test_save_path, file_name + ".jpg"))

        with open(os.path.join(test_save_path, os.path.split(json_path)[1]), "w") as outfile:
            json.dump(annotations, outfile, ensure_ascii=False, indent=4)

        f.close()


if __name__ == "__main__":
    read_write_data("hagrid/hagrid_all", "hagrid/temp", 100)