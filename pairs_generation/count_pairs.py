import glob
import sys


pairs_dir = "/home/acances/Data/TVHID/pairs16"


def nb_lines(file):
    i = -1
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


if __name__ == "__main__":
    pair_type = sys.argv[1]

    pairs_files = glob.glob("{}/{}/*".format(pairs_dir, pair_type))
    if pair_type == "positive":
        pairs_files = [e for e in pairs_files if e.split("/")[-1].split("_")[1] != "negative"]

    pairs = {}
    for file in pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs[video_id] = nb_lines(file)
    
    min_pairs = min(pairs.values())
    max_pairs = max(pairs.values())

    total_pairs = sum(pairs.values())
    nb_videos = len(pairs)

    # print("\t".join(map(str, sorted(pairs.values()))))

    print("Total:\t{}".format(total_pairs))
    print("Mean by file:\t{}".format(total_pairs/nb_videos))
    print("Min:\t{}".format(min_pairs))
    print("Max:\t{}".format(max_pairs))
