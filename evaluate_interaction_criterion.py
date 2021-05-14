import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle


GT_dir = "/home/acances/Data/TVHID/tv_human_interactions_annotations"
tracks_dir = "/home/acances/Data/TVHID/tracks"
track_linking_dir = "/home/acances/Data/TVHID/GT_tracks_to_det_tracks"

#  PARAMETERS
SEGMENT_LENGTH = 16
TEMP_INTERSECTION_THRESHOLD = SEGMENT_LENGTH
IOU_THRESHOLD = 0.2
FRAME_PROPORTION = 0.1


def get_interactions(video_id):
    # Each element in the returned vector "interactions" is of the form
    # [id_1, id_2, b, e]
    # where id_1 and id_2 are the track ids of the humans interacting (relative to the GT annotation files)
    # and b and e are the starting frame and ending frame (included) of the interaction,
    # starting with frame 0
    GT_file = "{}/{}.annotations".format(GT_dir, video_id)
    interactions = []
    with open(GT_file, "r") as f:
        f.readline()
        interacting = False
        previous_frame_id = None

        for line in f:
            if line[:6] == "#frame":
                entries = line.strip().split()
                frame_id = int(entries[1])
                if len(entries) > 4:
                    id_1 = int(entries[5])
                    id_2 = int(entries[7])
                    if not interacting:
                        interacting = True
                        interactions.append([id_1, id_2, frame_id])
                    elif [id_1, id_2] != interactions[-1][:2]:
                        interactions[-1].append(previous_frame_id)
                        interactions.append([id_1, id_2, frame_id])
                else:
                    if interacting:
                        interacting = False
                        interactions[-1].append(previous_frame_id)
                previous_frame_id = frame_id
        if interacting:
            interacting = False
            interactions[-1].append(previous_frame_id)
            
    for elt in interactions:
        assert len(elt) == 4
    return interactions


def get_tracks(video_id):
    tracks_files = sorted(glob.glob("{}/{}/*".format(tracks_dir, video_id)))
    tracks = []
    for tracks_file in tracks_files:
        with open(tracks_file, "rb") as f:
            tracks += pickle.load(f)
    tracks = [e[0] for e in tracks]
    return tracks


def get_GT_to_det(video_id):
    GT_to_det_file = "{}/{}_track_links.pkl".format(track_linking_dir, video_id)
    with open(GT_to_det_file, "rb") as f:
        GT_to_det = pickle.load(f)
    return GT_to_det


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    # b1 : [[x1, y1, x2, y2], ...]

    assert b1.shape == b2.shape

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2], b2[:,2])
    ymax = np.minimum(b1[:,3], b2[:,3])

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(tube1, tube2):
    """Compute the frame IoU vector of two tubes with the same temporal extent"""
    # tube1 : [[x1, y1, x2, y2], ...]
    
    assert tube1.shape[0] == tube2.shape[0]

    overlap = overlap2d(tube1, tube2)
    iou = overlap / (area2d(tube1) + area2d(tube2) - overlap)

    return iou


def is_segment_positive(segment, id_1, id_2, interactions):
    begin, end = segment
    for i1, i2, b, e in interactions:
        if (i1, i2) == (id_1, id_2):
            return begin <= e and b <= end
    return False


def is_segment_considered_positive(segment, track_1, track_2):
    begin, end = segment

    b1 = int(track_1[0, 0])
    b2 = int(track_2[0, 0])

    tube_1 = track_1[begin - b1:end - b1, 1:5]
    tube_2 = track_2[begin - b2:end - b2, 1:5]

    iou = iou2d(tube_1, tube_2)
    is_above = iou > IOU_THRESHOLD

    return np.sum(is_above) / SEGMENT_LENGTH >= FRAME_PROPORTION


def is_segment_considered_negative(segment, track_1, track_2):
    begin, end = segment

    b1 = int(track_1[0, 0])
    b2 = int(track_2[0, 0])

    tube_1 = track_1[begin - b1:end - b1, 1:5]
    tube_2 = track_2[begin - b2:end - b2, 1:5]
    
    iou = iou2d(tube_1, tube_2)
    is_above = iou > IOU_THRESHOLD

    return np.sum(is_above) == 0


def get_segments(tracks, GT_to_det):
    segments = {}
    for id_1 in sorted(GT_to_det.keys()):
        for id_2 in sorted(GT_to_det.keys()):
            if id_2 <= id_1:
                continue

            track_1 = tracks[GT_to_det[id_1][0]]
            track_2 = tracks[GT_to_det[id_2][0]]

            begin = int(max(track_1[0, 0], track_2[0, 0]))
            end = int(min(track_1[-1, 0], track_2[-1, 0]))

            segment_begin_indices = list(range(begin, end - SEGMENT_LENGTH + 2, SEGMENT_LENGTH))
            for begin_frame in segment_begin_indices:
                if (id_1, id_2) not in segments:
                    segments[(id_1, id_2)] = []
                segment = [begin_frame, begin_frame + SEGMENT_LENGTH]
                segments[(id_1, id_2)].append(segment)
    return segments


def get_statistics(video_id):
    interactions = get_interactions(video_id)
    tracks = get_tracks(video_id)
    GT_to_det = get_GT_to_det(video_id)

    TP = 0
    FP = 0

    TN = 0
    FN = 0

    IP = 0      # ignored positives
    IN = 0      # ignored negatives

    segments = get_segments(tracks, GT_to_det)
    for id_1, id_2 in segments:
        track_1 = tracks[GT_to_det[id_1][0]]
        track_2 = tracks[GT_to_det[id_2][0]]
        for segment in segments[(id_1, id_2)]:
            positive = is_segment_positive(segment, id_1, id_2, interactions)
            considered_positive = is_segment_considered_positive(segment, track_1, track_2)
            considered_negative = is_segment_considered_negative(segment, track_1, track_2)

            if positive:
                if considered_positive:
                    TP += 1
                elif considered_negative:
                    FN += 1
                else:
                    IP += 1
            else:
                if considered_positive:
                    FP += 1
                elif considered_negative:
                    TN += 1
                else:
                    IN += 1

    return [TP, FP, TN, FN, IP, IN]


def compute_statistics():
    # TP, FP, TN, FN, IP, IN
    statistics = [0 for i in range(6)]

    video_ids = [e.split("/")[-1].split(".")[0] for e in glob.glob("{}/*".format(GT_dir))]
    for video_id in video_ids:
        video_stats = get_statistics(video_id)
        for i in range(6):
            statistics[i] += video_stats[i]
    
    TP, FP, TN, FN, IP, IN = statistics
    
    print("TP: \t {} \t FP: \t {} \t IP: \t {}".format(TP, FP, IP))
    print("TN: \t {} \t FN: \t {} \t IN: \t {}".format(TN, FN, IN))

    with open("{}/stats_IoU{}_prop{}.csv".format(IOU_THRESHOLD, FRAME_PROPORTION), "w'") as f:
        f.write("{},{},{}\n".format(TP, FP, IP))
        f.write("{},{},{}\n".format(TN, FN, IN))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print("LA")
        IOU_THRESHOLD = float(sys.argv[1])
        FRAME_PROPORTION = float(sys.argv[2])
    # compute_statistics()
