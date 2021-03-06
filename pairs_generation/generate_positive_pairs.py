import cv2
import numpy as np
import pickle
import sys
import glob
from pathlib import Path


#  PARAMETERS
SEGMENT_LENGTH = 16
ACCEPTABLE_DELAY = 3 * SEGMENT_LENGTH // 4


GT_dir = "/home/acances/Data/TVHID/tv_human_interactions_annotations"
tracks_dir = "/home/acances/Data/TVHID/tracks"
track_linking_dir = "/home/acances/Data/TVHID/GT_tracks_to_det_tracks"
pairs_dir = "/home/acances/Data/TVHID/pairs{}".format(SEGMENT_LENGTH)


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


def generate_positive_pairs(video_id):
    interactions = get_interactions(video_id)
    tracks = get_tracks(video_id)
    GT_to_det = get_GT_to_det(video_id)

    # print("Interactions:")
    # for elt in interactions: print(elt)
    # print()

    pairs = []
    for id_1, id_2, b, e in interactions:
        if (id_1 not in GT_to_det) or (id_2 not in GT_to_det):
            continue
        track_1 = tracks[GT_to_det[id_1][0]]
        track_2 = tracks[GT_to_det[id_2][0]]

        inter_b = int(max(track_1[0, 0], track_2[0, 0]))
        inter_e = int(min(track_1[-1, 0], track_2[-1, 0]))

        begin = max(b - ACCEPTABLE_DELAY, inter_b)
        end = min(e + ACCEPTABLE_DELAY, inter_e)

        segment_begin_indices = list(range(begin, end - SEGMENT_LENGTH, SEGMENT_LENGTH))
        for begin_frame in segment_begin_indices:
            pair = []
            pair += [video_id, GT_to_det[id_1][0], begin_frame, begin_frame + SEGMENT_LENGTH]
            pair += [video_id, GT_to_det[id_2][0], begin_frame, begin_frame + SEGMENT_LENGTH]
            pairs.append(pair)

        # print("tracks {} - {} : \t".format(id_1, id_2) + "\t".join(map(str, segment_begin_indices)))
    
    output_file = "{}/positive/pairs_{}.csv".format(pairs_dir, video_id)
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(",".join(map(str, pair)) + "\n")


if __name__ == "__main__":
    video_ids = [e.split("/")[-1] for e in sorted(glob.glob("{}/*".format(tracks_dir)))]
    for video_id in video_ids:
        generate_positive_pairs(video_id)

    # video_id = sys.argv[1]
    # generate_positive_pairs(video_id)
