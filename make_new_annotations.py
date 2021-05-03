import numpy as np
import subprocess
import glob
from pathlib import Path
import tqdm
from read_original_annotation_file import read_original_annotation_file



def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]+1) * (b[:,3]-b[:,1]+1)

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2] + 1, b2[:,2] + 1)
    ymax = np.minimum(b1[:,3] + 1, b2[:,3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def IoUB(UB_box, boxes):
    """Compute the intersections over upper body area (IoUB) for an upper body box UB_box and a set of boxes boxes"""
    overlaps = overlap2d(boxes, UB_box)
    UB_area = area2d(UB_box)
    IoUB = overlaps / UB_area
    return IoUB


def write_frame_info(boxes_by_frame, new_annotation_file):
    nb_frames = len(boxes_by_frame)
    with open(new_annotation_file, "w") as f:
        f.write(str(nb_frames) + "\n")
        for frame_id, frame_boxes in enumerate(boxes_by_frame):
            f.write("frame " + str(frame_id + 1) + "\n")
            for box_info in frame_boxes:
                box_coords = box_info[0]
                interacting = box_info[1]
                entries = box_coords + (int(interacting), )
                f.write("\t".join(map(str, entries)))
                f.write("\n")


def make_new_annotations():
    annotation_files = sorted(glob.glob("/home/acances/Data/TVHID/tv_human_interactions_annotations/*"))

    for annotation_file in annotation_files:
        video_name = annotation_file.split("/")[-1].split(".")[0]
        boxes_by_frame = read_original_annotation_file(annotation_file)

        new_boxes_by_frame = []

        not_enough_detectron2_boxes = False
        for frame_id, frame_boxes in enumerate(boxes_by_frame):
            detectron2_box_file = "/home/acances/Data/TVHID/detectron2_boxes/{}/{:06d}_boxes.npy".format(video_name, frame_id + 1)
            detectron2_boxes = np.load(detectron2_box_file)

            new_frame_boxes = []

            not_enough_detectron2_boxes = (len(detectron2_boxes) < len(frame_boxes))
            if not_enough_detectron2_boxes:
                break

            for UB_box_info in frame_boxes:
                UB_box = np.array(UB_box_info[0]).reshape(1, -1)
                interacting = UB_box_info[1]

                IoUBs = IoUB(UB_box, detectron2_boxes)
                max_id = np.argmax(IoUBs)
                new_box_info = [tuple(detectron2_boxes[max_id]), interacting]
                new_frame_boxes.append(new_box_info)

                detectron2_boxes = np.delete(detectron2_boxes, max_id, axis=0)

            new_boxes_by_frame.append(new_frame_boxes)
        
        if not_enough_detectron2_boxes:
            print(video_name)
            continue

        new_annotation_file = "/home/acances/Data/TVHID/new_annotations/{}_annotations.txt".format(video_name)
        write_frame_info(new_boxes_by_frame, new_annotation_file)


if __name__ =="__main__":
    make_new_annotations()
