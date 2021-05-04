import cv2
import numpy as np
import pickle
import sys
import glob
from pathlib import Path
import random


#  PARAMETERS
SEGMENT_LENGTH = 16
MAX_PAIRS_BY_VIDEO = 2
NB_TRIES_BY_VIDEO = 100


# Fix seed for reproducibility
random.seed(0)


GT_dir = "/home/acances/Data/TVHID/tv_human_interactions_annotations"
tracks_dir = "/home/acances/Data/TVHID/tracks"
track_linking_dir = "/home/acances/Data/TVHID/GT_tracks_to_det_tracks"
pairs_dir = "/home/acances/Data/TVHID/pairs{}".format(SEGMENT_LENGTH)


def get_interactions(video_id):
    # Each element in the returned vector "interactions" si of the form
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


def generate_negative_pairs(video_id):
    interactions = get_interactions(video_id)
    tracks = get_tracks(video_id)
    GT_to_det = get_GT_to_det(video_id)

    interacting_pairs_of_ids = [e[:2] for e in interactions]

    # print("Interactions:")
    # for elt in interactions: print(elt)
    # print()

    pairs = []
    if len(GT_to_det) >= 2:
        pairs_of_ids_used = []
        for i in range(NB_TRIES_BY_VIDEO):
            id_1, id_2 = sorted(random.sample(GT_to_det.keys(), 2))
            if [id_1, id_2] in interacting_pairs_of_ids:
                continue
            if [id_1, id_2] in pairs_of_ids_used:
                continue

            track_1 = tracks[GT_to_det[id_1][0]]
            track_2 = tracks[GT_to_det[id_2][0]]

            inter_b = int(max(track_1[0, 0], track_2[0, 0]))
            inter_e = int(min(track_1[-1, 0], track_2[-1, 0]))
            if inter_e - inter_b + 1 < SEGMENT_LENGTH:
                continue

            begin_frame = random.randint(inter_b, inter_e - SEGMENT_LENGTH + 1)

            pair = []
            pair += [video_id, GT_to_det[id_1][0], begin_frame, begin_frame + SEGMENT_LENGTH]
            pair += [video_id, GT_to_det[id_2][0], begin_frame, begin_frame + SEGMENT_LENGTH]
            pairs.append(pair)

            if len(pairs) == MAX_PAIRS_BY_VIDEO:
                break

    output_file = "{}/negative/pairs_{}.csv".format(pairs_dir, video_id)
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(",".join(map(str, pair)) + "\n")


if __name__ == "__main__":
    video_ids = [e.split("/")[-1] for e in sorted(glob.glob("{}/*".format(tracks_dir)))]
    for video_id in video_ids:
        generate_negative_pairs(video_id)

    # video_id = sys.argv[1]
    # generate_negative_pairs(video_id)
