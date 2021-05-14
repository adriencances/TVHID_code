import os
import sys
import numpy as np
import cv2
import glob
import tqdm
from pathlib import Path
import pickle


def gather_results():
    results_dir = "/home/acances/Code/TVHID/interaction_criterion_evaluation"
    output_file = results_dir + "/stats_summary.csv"

    results = {}

    labels = ["total", "handShake", "highFive", "hug","kiss", "negative"]
    results_files = glob.glob(results_dir + "/stats_IoU*.csv")
    for file in results_files:
        iou = float(file.split("/")[-1].split("_")[1][3:])
        prop = float(file.split("/")[-1].split("_")[2][4:-4])

        statistics = {}
        percentages = {}
        with open(file, "r") as f:
            for label in labels:
                statistics[label] = f.readline().strip().split(",")
            for label in labels:
                percentages[label] = f.readline().strip().split(",")

        results[(prop, iou)] = (statistics, percentages)
    
    with open(output_file, "w") as f:
        f.write(",".join(["iou", "prop"]))
        for label in labels:
            f.write("," + ",".join([label, "TP", "FP", "IP", "TN", "FN", "IN"]))
        f.write("\n")
        for prop, iou in sorted(results.keys()):
            f.write(",".join(map(str, [iou, prop])))
            statistics, percentages = results[(prop, iou)]
            for label in statistics:
                f.write("," + ",".join(map(str, [""] + statistics[label])))
            f.write("\n")
        f.write(" , \n")

        f.write(",".join(["iou", "prop"]))
        for label in labels:
            f.write("," + ",".join([label, "TP", "FP", "IP", "TN", "FN", "IN"]))
        f.write("\n")
        for prop, iou in sorted(results.keys()):
            f.write(",".join(map(str, [iou, prop])))
            statistics, percentages = results[(prop, iou)]
            for label in percentages:
                f.write("," + ",".join(map(str, [""] + percentages[label])))
            f.write("\n")


if __name__ == "__main__":
    gather_results()
