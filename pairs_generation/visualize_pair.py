import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m
from torchvision import transforms, utils
import cv2
import sys
import pickle
import tqdm
import glob
import os
from pathlib import Path


class FrameProcessor:
    def __init__(self, w, h, alpha, frames_dir, tracks_dir, normalized_boxes=False):
        # (w, h) : dimensions of processed frame
        # alpha : quantity by which the bounding box areas get enlarged
        self.w = w
        self.h = h
        self.alpha = alpha
        self.normalized_boxes = normalized_boxes

        self.frames_dir = frames_dir
        self.tracks_dir = tracks_dir


    def enlarged_box(self, box):
        # Enlarge the box area by 100*alpha percent while preserving
        # the center and the aspect ratio
        beta = 1 + self.alpha
        x1, y1, x2, y2 = box
        dx = x2 - x1
        dy = y2 - y1
        x1 -= (np.sqrt(beta) - 1)*dx/2
        x2 += (np.sqrt(beta) - 1)*dx/2
        y1 -= (np.sqrt(beta) - 1)*dy/2
        y2 += (np.sqrt(beta) - 1)*dy/2
        return x1, y1, x2, y2

    def preprocessed_frame(self, video_id, n):
        # n : frame index in the timestamp (frame indices start at 1)
        frame_file = "{}/{}/{:06d}.jpg".format(self.frames_dir, video_id, n)
        assert os.path.isfile(frame_file), frame_file
        # frame : H * W * 3
        frame = cv2.imread(frame_file)
        # frame : 3 * W * H
        frame = frame.transpose(2, 1, 0)
        frame = torch.from_numpy(frame)
        return frame

    def processed_frame(self, frame, box):
        # frame : 3 * W * H
        # (w, h) : dimensions of new frame

        C, W, H = frame.shape
        x1, y1, x2, y2 = box

        # If box is in normalized coords, i.e.
        # image top-left corner (0,0), bottom-right (1, 1),
        # then turn normalized coord into absolute coords
        if self.normalized_boxes:
            x1 = x1*W
            x2 = x2*W
            y1 = y1*H
            y2 = y2*H

        # Round coords to integers
        X1 = max(0, m.floor(x1))
        X2 = max(0, m.ceil(x2))
        Y1 = max(0, m.floor(y1))
        Y2 = max(0, m.ceil(y2))
        
        dX = X2 - X1
        dY = Y2 - Y1

        # Get the cropped bounding box
        boxed_frame = transforms.functional.crop(frame, X1, Y1, dX, dY)
        dX, dY = boxed_frame.shape[1:]

        # Compute size to resize the cropped bounding box to
        if dY/dX >= self.h/self.w:
            w_tild = m.floor(dX/dY*self.h)
            h_tild = self.h
        else:
            w_tild = self.w
            h_tild = m.floor(dY/dX*self.w)
        assert w_tild <= self.w
        assert h_tild <= self.h

        # Get the resized cropped bounding box
        resized_boxed_frame = transforms.functional.resize(boxed_frame, [w_tild, h_tild])

        # Put the resized cropped bounding box on a gray canvas
        new_frame = 127*torch.ones(C, self.w, self.h)
        i = m.floor((self.w - w_tild)/2)
        j = m.floor((self.h - h_tild)/2)
        new_frame[:, i:i+w_tild, j:j+h_tild] = resized_boxed_frame
        return new_frame

    def track(self, video_id, track_id):
        tracks_files = sorted(glob.glob("{}/{}/*".format(self.tracks_dir, video_id)))
        tracks = []
        for tracks_file in tracks_files:
            with open(tracks_file, "rb") as f:
                tracks += pickle.load(f)
        tracks = [e[0] for e in tracks]
        return tracks[track_id]

    def processed_frames(self, video_id, track_id, begin_frame, end_frame):
        # begin_frame, end_frame : 0-based indices

        track = self.track(video_id, track_id)
        b = int(track[0, 0])

        processed_frames = []
        for i in range(begin_frame, end_frame):
            frame = self.preprocessed_frame(video_id, i + 1)
            track_frame_index = i - b
            box = track[track_frame_index][1:5]
            box = self.enlarged_box(box)
            processed_frame = self.processed_frame(frame, box)
            processed_frames.append(processed_frame)
        processed_frames = torch.stack(processed_frames, dim=1)
        return processed_frames


class Printer:
    def __init__(self):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.frames_dir = "/home/acances/Data/TVHID/keyframes"
        self.tracks_dir = "/home/acances/Data/TVHID/tracks"
        self.pairs_dir = "/home/acances/Data/TVHID/pairs16"
        self.features_dir = "/home/acances/Data/TVHID/features16"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.frames_dir, self.tracks_dir)

        self.output_dir = "/home/acances/Data/TVHID/pairs_images"


    def get_tensors(self, pair):
        video_id1, track_id1, begin1, end1, video_id2, track_id2, begin2, end2 = pair

        track_id1, begin1, end1 = list(map(int, [track_id1, begin1, end1]))
        track_id2, begin2, end2 = list(map(int, [track_id2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, track_id2, begin2, end2)

        return tensor1, tensor2

    def print_pair(self, pairs_file, line_nb):
        with open(pairs_file, "r") as f:
            for i in range(line_nb):
                f.readline()
            pair = f.readline().strip().split(",")
        tensor1, tensor2 = self.get_tensors(pair)    # 3x16x224x244

        category = pairs_file.split("/")[-2]
        video_id = pairs_file.split("/")[-1].split(".")[0][6:]
        subdir = "{}/{}/{}/pair_{}".format(self.output_dir, category, video_id, line_nb)
        Path(subdir).mkdir(parents=True, exist_ok=True)

        for i in range(tensor1.shape[1]):
            filename1 = "{}/tensor1_frame_{}.jpg".format(subdir, i + 1)
            frame1 = tensor1[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename1, frame1)

            filename2 = "{}/tensor2_frame_{}.jpg".format(subdir, i + 1)
            frame2 = tensor2[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename2, frame2)
    
    def get_nb_lines(self, file):
        i = -1
        with open(file, "r") as f:
            for i, f in enumerate(f): pass
        return i + 1

    def print_pairs(self, pairs_file):
        nb_lines = self.get_nb_lines(pairs_file)
        for line_nb in range(nb_lines):
            self.print_pair(pairs_file, line_nb)


if __name__ == "__main__":
    printer = Printer()

    pairs_files = sys.argv[1:]
    for pairs_file in tqdm.tqdm(pairs_files):
        printer.print_pairs(pairs_file)
