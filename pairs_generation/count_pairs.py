import glob
import sys


pairs_dir = "/home/acances/Data/TVHID/pairs16"


def nb_lines(file):
    i = -1
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


def count_by_classes():
    pairs_files = glob.glob("{}/positive/*".format(pairs_dir))
    pairs_files = [e for e in pairs_files if e.split("/")[-1].split("_")[1] != "negative"]

    pairs_by_classes = {}
    for file in pairs_files:
        label = file.split("/")[-1].split("_")[1]
        if label not in pairs_by_classes:
            pairs_by_classes[label] = {}
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs_by_classes[label][video_id] = nb_lines(file)
    
    for label in pairs_by_classes:
        pairs = pairs_by_classes[label]
        min_pairs = min(pairs.values())
        max_pairs = max(pairs.values())
        total_pairs = sum(pairs.values())
        nb_videos = len(pairs)
        print(label)
        print(
            "Total: {} \t Mean by file: {} \t Min: {} \t Max: {}".format(total_pairs, total_pairs/nb_videos, min_pairs, max_pairs)
        )
        print(" ".join(map(str, sorted(pairs.values()))))


def count_by_type(pairs_type):
    pairs_files = glob.glob("{}/{}/*".format(pairs_dir, pairs_type))
    if pairs_type == "positive":
        pairs_files = [e for e in pairs_files if e.split("/")[-1].split("_")[1] != "negative"]

    pairs = {}
    for file in pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs[video_id] = nb_lines(file)
    
    min_pairs = min(pairs.values())
    max_pairs = max(pairs.values())

    total_pairs = sum(pairs.values())
    nb_videos = len(pairs)

    print(" ".join(map(str, sorted(pairs.values()))))

    print("Total:\t{}".format(total_pairs))
    print("Mean by file:\t{}".format(total_pairs/nb_videos))
    print("Min:\t{}".format(min_pairs))
    print("Max:\t{}".format(max_pairs))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pairs_type = sys.argv[1]
        count_by_type(pairs_type)
    else:
        count_by_classes()

    
