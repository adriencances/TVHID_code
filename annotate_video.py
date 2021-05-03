import cv2
import numpy as np
import pickle
import sys
import glob
from pathlib import Path


tracks_dir = "/home/acances/Data/TVHID/tracks"
frames_dir = "/home/acances/Data/TVHID/keyframes"
new_frames_dir = "/home/acances/Data/TVHID/annotated_frames"


def annotate_shot(video_id):
    Path("{}/{}".format(new_frames_dir, video_id)).mkdir(parents=True, exist_ok=True)

    tracks_files = glob.glob("{}/{}/*".format(tracks_dir, video_id))
    tracks = []
    for tracks_file in tracks_files:
        with open(tracks_file, "rb") as f:
            tracks += pickle.load(f)

    frame_files = sorted(glob.glob("{}/{}/*".format(frames_dir, video_id)))
    nb_frames = len(frame_files)

    boxes_by_frame = dict([(frame_id, []) for frame_id in range(nb_frames + 1)])
    for track in tracks:
        for box_info in track[0]:
            frame_id = int(box_info[0])
            box = list(box_info[1:5])
            boxes_by_frame[frame_id].append(box)

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        frame_id = int(frame_file.split("/")[-1].split(".")[0]) - 1
        boxes = boxes_by_frame[frame_id]
        color = (0, 255, 0)
        thickness = 2
        for box in boxes:
            box = tuple(map(int, list(box[:4])))
            img = cv2.rectangle(img, box[:2], box[2:], color, thickness)
        new_frame_file = "{}/{}/{:06d}.jpg".format(new_frames_dir, video_id, frame_id + 1)
        cv2.imwrite(new_frame_file, img)


if __name__ == "__main__":
    video_id = sys.argv[1]
    annotate_shot(video_id)



