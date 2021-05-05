import glob
import sys


pairs_dir = "/home/acances/Data/TVHID/pairs16"
train_file = "/home/acances/Data/TVHID/split/train.txt"
val_file = "/home/acances/Data/TVHID/split/val.txt"


def nb_lines(file):
    i = -1
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


def split_positives():
    pairs_files = glob.glob("{}/positive/*".format(pairs_dir))
    pairs_files = [e for e in pairs_files if e.split("/")[-1].split("_")[1] != "negative"]

    all_pairs = {}
    pairs_by_classes = {}
    for file in pairs_files:
        label = file.split("/")[-1].split("_")[1]
        if label not in pairs_by_classes:
            pairs_by_classes[label] = {}
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs_by_classes[label][video_id] = nb_lines(file)
        all_pairs[video_id] = nb_lines(file)
    
    for label in pairs_by_classes:
        for video_id in list(pairs_by_classes[label]):
            if pairs_by_classes[label][video_id] == 0:
                del pairs_by_classes[label][video_id]
        # print(" ".join(map(str, sorted(pairs_by_classes[label].values()))))
    
    train_video_ids = []
    val_video_ids = []

    for label in pairs_by_classes:
        pairs = pairs_by_classes[label]
        sorted_video_ids = sorted(pairs.keys(), key=lambda video_id: pairs[video_id], reverse=True)
        # print(" ".join(map(str, [pairs[video_id] for video_id in sorted_video_ids])))
        for i, video_id in enumerate(sorted_video_ids):
            if i%4 == 0:
                val_video_ids.append(video_id)
            else:
                train_video_ids.append(video_id)
        print(label)
        print("\ttrain: \t {}".format(sum([pairs[video_id] for video_id in sorted_video_ids if video_id in train_video_ids])))
        print("\tval: \t {}".format(sum([pairs[video_id] for video_id in sorted_video_ids if video_id in val_video_ids])))
    
    nb_train = sum([all_pairs[video_id] for video_id in train_video_ids])
    nb_val = sum([all_pairs[video_id] for video_id in val_video_ids])
    print("total for positives")
    print("\ttrain: \t {}".format(nb_train))
    print("\tval: \t {}".format(nb_val))

    return train_video_ids, val_video_ids


def split_negatives():
    pairs_files = glob.glob("{}/negative/*".format(pairs_dir))
    pairs_files = [e for e in pairs_files if e.split("/")[-1].split("_")[1] == "negative"]

    pairs = {}
    for file in pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs[video_id] = nb_lines(file)
    
    train_video_ids = []
    val_video_ids = []

    sorted_video_ids = sorted(pairs.keys(), key=lambda video_id: pairs[video_id], reverse=True)
    for i, video_id in enumerate(sorted_video_ids):
        if i%4 == 0:
            val_video_ids.append(video_id)
        else:
            train_video_ids.append(video_id)
    
    return train_video_ids, val_video_ids


def split_data():
    pos_train_video_ids, pos_val_video_ids = split_positives()
    neg_train_video_ids, neg_val_video_ids = split_negatives()

    train_video_ids = pos_train_video_ids + neg_train_video_ids
    val_video_ids = pos_val_video_ids + neg_val_video_ids

    pairs_files = glob.glob("{}/negative/*".format(pairs_dir))
    pairs = {}
    for file in pairs_files:
        video_id = file.split("/")[-1].split(".")[0][6:]
        pairs[video_id] = nb_lines(file)
    nb_train_neg = sum([pairs[video_id] for video_id in train_video_ids])
    nb_val_neg = sum([pairs[video_id] for video_id in val_video_ids])
    print("total for negatives")
    print("\ttrain: \t {}".format(nb_train_neg))
    print("\tval: \t {}".format(nb_val_neg))

    with open(train_file, "w") as f:
        for video_id in train_video_ids:
            f.write(video_id + "\n")
    
    with open(val_file, "w") as f:
        for video_id in val_video_ids:
            f.write(video_id + "\n")


if __name__ == "__main__":
    split_data()

    
