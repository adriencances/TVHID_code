import numpy as np
import cv2
from read_new_annotation_file import read_new_annotation_file
import glob
import sys



if __name__ == "__main__":
    frames_dir = sys.argv[1]
    new_frames_dir = sys.argv[2]
    annotation_file = sys.argv[3]

    boxes_by_frames = read_new_annotation_file(annotation_file)
    nb_frames = len(boxes_by_frames)

    frame_files = sorted(glob.glob(frames_dir + "/*"))

    for frame_id, frame_boxes in enumerate(boxes_by_frames):
        frame_file = frame_files[frame_id]
        frame_name = frame_file.split("/")[-1].split(".")[0]
        new_frame_file = new_frames_dir + "/" + frame_name + "_annotated.jpg"

        img = cv2.imread(frame_file)
        colors = [(0, 255, 0), (0, 0, 255)]
        thickness = 2
        for box_info in frame_boxes:
            box = box_info[0]
            interacting = box_info[1]
            img = cv2.rectangle(img, box[:2], box[2:], colors[interacting], thickness)
        cv2.imwrite(new_frame_file, img)








