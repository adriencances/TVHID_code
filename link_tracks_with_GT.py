import cv2
import numpy as np
import pickle
import sys
import glob
from pathlib import Path


GT_dir = "/home/acances/Data/TVHID/tv_human_interactions_annotations"
tracks_dir = "/home/acances/Data/TVHID/tracks"
track_linking_dir = "/home/acances/Data/TVHID/GT_tracks_to_det_tracks"


def get_GT_tracks(GT_file):
    with open(GT_file, "r") as f:
        # First line with number of frames
        f.readline()
        # frame_id of first frame is 0
        frame_id = None
        tracks = []
        for line in f:
            if line[:6] == "#frame":
                frame_id = int(line.strip().split()[1])
                continue
            track_id, x1, y1, size = tuple(map(int, line.split()[:4]))
            box = [x1, y1, x1 + size, y1 + size]
            while track_id >= len(tracks):
                tracks.append({})
            tracks[track_id][frame_id] = box
    
    return tracks
            

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)

    return width * height


def intersection_over_GT_area(det_box, GT_box):
    GT_area = area(GT_box)
    intersection = intersection_area(det_box, GT_box)
    return intersection / GT_area


def mean_IoGT(det_track, GT_track):
    assert len(det_track) > 0
    IoGTs = [0 for i in range(len(det_track))]
    for i in range(len(det_track)):
        frame_id = int(det_track[i][0])
        det_box = det_track[i][1:5]
        if frame_id not in GT_track:
            continue
        GT_box = GT_track[frame_id]
        IoGTs[i] = intersection_over_GT_area(det_box, GT_box)
    return sum(IoGTs) / len(IoGTs)


def time_IoGT(det_track, GT_track):
    det_b = det_track[0, 0]
    det_e = det_track[-1, 0]
    GT_b = min(GT_track.keys())
    GT_e = max(GT_track.keys())

    b = max(det_b, GT_b)
    e = min(det_e, GT_e)
    inter = max(0, e - b + 1)

    return inter / len(GT_track)


def link_tracks(video_id):
    tracks_files = sorted(glob.glob("{}/{}/*".format(tracks_dir, video_id)))
    det_tracks = []
    for tracks_file in tracks_files:
        with open(tracks_file, "rb") as f:
            det_tracks += pickle.load(f)
    det_scores = [e[1] for e in det_tracks]
    det_tracks = [e[0] for e in det_tracks]

    GT_file = "{}/{}.annotations".format(GT_dir, video_id)
    GT_tracks = get_GT_tracks(GT_file)

    time_IoGTs = []
    GT_to_det = {}
    for det_track_id, det_track in enumerate(det_tracks):
        if len(GT_to_det) == len(GT_tracks):
            break
        mean_IoGTs = [mean_IoGT(det_track, GT_track) for GT_track in GT_tracks]
        indices = list(reversed(np.argsort(mean_IoGTs)))
        best_GT_track_id = indices[0]
        if mean_IoGTs[best_GT_track_id] < 0.4:
            continue
        if best_GT_track_id not in GT_to_det:
            GT_to_det[best_GT_track_id] = (det_track_id, mean_IoGTs[best_GT_track_id])
            time_IoGTs.append(time_IoGT(det_track, GT_tracks[best_GT_track_id]))

        # print(mean_IoGTs, det_scores[det_track_id])
        # print(list(reversed(np.argsort(mean_IoGTs))))
        # print(np.argmax(mean_IoGTs))
    
    output_file = "{}/{}_track_links.pkl".format(track_linking_dir, video_id)
    with open(output_file, "wb") as f:
        pickle.dump(GT_to_det, f)

    # print(np.mean(time_IoGTs))
    # # for key in GT_to_det:
    # #     print(GT_to_det[key])
    # if len(GT_to_det) != len(GT_tracks):
    #     # print("INCOMPLETE")
    #     print(video_id)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        video_id = sys.argv[1]
        link_tracks(video_id)
    else:
        video_ids = [e.split("/")[-1] for e in sorted(glob.glob("{}/*".format(tracks_dir)))]
        for video_id in video_ids:
            link_tracks(video_id)
