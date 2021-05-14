import glob
import sys


tracks_dir = "/home/acances/Data/TVHID/tracks"
pairs_dir = "/home/acances/Data/TVHID/pairs16"
train_file = "/home/acances/Data/TVHID/split/train.txt"
val_file = "/home/acances/Data/TVHID/split/val.txt"
test_file = "/home/acances/Data/TVHID/split/test.txt"


def get_set1_set2():
    set1 = {}
    set1["handShake"] = [2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    set1["highFive"] = [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48]
    set1["hug"] = [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50]
    set1["kiss"] = [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]
    set1["negative"] = list(range(1,51))

    set2 = {}
    set2["handShake"] = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39]
    set2["highFive"] = [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50]
    set2["hug"] = [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48]
    set2["kiss"] = [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]
    set2["negative"] = list(range(51,101))

    return set1, set2


def nb_lines(file):
    i = -1
    with open(file, "r") as f:
        for i, line in enumerate(f): pass
    return i + 1


def split_positives(set):
    all_pairs = {}
    pairs_by_classes = {}
    labels = [label for label in set if label != "negative"]
    for label in labels:
        pairs_by_classes[label] = {}
        for i in set[label]:
            video_id = "{}_{:04d}".format(label, i)
            pairs_file = "{}/positive/pairs_{}.csv".format(pairs_dir, video_id)

            all_pairs[video_id] = nb_lines(pairs_file)
            pairs_by_classes[label][video_id] = all_pairs[video_id]

    for label in pairs_by_classes:
        for video_id in list(pairs_by_classes[label]):
            if pairs_by_classes[label][video_id] == 0:
                del pairs_by_classes[label][video_id]
    
    train_video_ids = []
    val_video_ids = []

    for label in pairs_by_classes:
        pairs = pairs_by_classes[label]
        sorted_video_ids = sorted(pairs.keys(), key=lambda video_id: pairs[video_id], reverse=True)
        for i, video_id in enumerate(sorted_video_ids):
            if i%4 == 0:
                val_video_ids.append(video_id)
            else:
                train_video_ids.append(video_id)

    return train_video_ids, val_video_ids


def split_negatives(set):
    video_ids = ["negative_{:04d}".format(i) for i in set["negative"]]

    pairs = {}
    for video_id in video_ids:
        pairs_file = "{}/negative/pairs_{}.csv".format(pairs_dir, video_id)
        pairs[video_id] = nb_lines(pairs_file)
    
    for video_id in list(pairs):
        if pairs[video_id] == 0:
            del pairs[video_id]
    
    train_video_ids = []
    val_video_ids = []

    sorted_video_ids = sorted(pairs.keys(), key=lambda video_id: pairs[video_id], reverse=True)
    for i, video_id in enumerate(sorted_video_ids):
        if i%4 == 0:
            val_video_ids.append(video_id)
        else:
            train_video_ids.append(video_id)
    
    return train_video_ids, val_video_ids


def get_test_video_ids(set):
    test_video_ids = []

    all_pairs = {}
    pairs_by_classes = {}
    for label in set:
        pairs_by_classes[label] = {}
        for i in set[label]:
            video_id = "{}_{:04d}".format(label, i)
            pairs_file = "{}/positive/pairs_{}.csv".format(pairs_dir, video_id)

            all_pairs[video_id] = nb_lines(pairs_file)
            pairs_by_classes[label][video_id] = all_pairs[video_id]

            if all_pairs[video_id] != 0:
                test_video_ids.append(video_id)
    
    return test_video_ids


def print_stats(video_ids):
    labels = ["handShake", "highFive", "hug", "kiss", "negative"]
    pos_pairs_by_classes = {}
    neg_pairs_by_classes = {}
    for label in labels:
        pos_pairs_by_classes[label] = {}
        neg_pairs_by_classes[label] = {}
    for video_id in video_ids:
        label = video_id.split("_")[0]
        pos_pairs_file = "{}/positive/pairs_{}.csv".format(pairs_dir, video_id)
        neg_pairs_file = "{}/negative/pairs_{}.csv".format(pairs_dir, video_id)

        pos_pairs_by_classes[label][video_id] = nb_lines(pos_pairs_file)
        neg_pairs_by_classes[label][video_id] = nb_lines(neg_pairs_file)
    
    print("\t".join(map(str, [sum(pos_pairs_by_classes[label].values()) for label in labels])), end="\t\t")
    print(sum([sum(pos_pairs_by_classes[label].values()) for label in labels]))
    print("\t".join(map(str, [sum(neg_pairs_by_classes[label].values()) for label in labels])), end="\t\t")
    print(sum([sum(neg_pairs_by_classes[label].values()) for label in labels]))


def split_data():
    set1, set2 = get_set1_set2()

    pos_train_video_ids, pos_val_video_ids = split_positives(set1)
    neg_train_video_ids, neg_val_video_ids = split_negatives(set1)

    train_video_ids = pos_train_video_ids + neg_train_video_ids
    val_video_ids = pos_val_video_ids + neg_val_video_ids

    test_video_ids = get_test_video_ids(set2)

    with open(train_file, "w") as f:
        for video_id in train_video_ids:
            f.write(video_id + "\n")
    
    with open(val_file, "w") as f:
        for video_id in val_video_ids:
            f.write(video_id + "\n")
    
    with open(test_file, "w") as f:
        for video_id in test_video_ids:
            f.write(video_id + "\n")

    print("train")
    print_stats(train_video_ids)
    print("val")
    print_stats(val_video_ids)
    print("test")
    print_stats(test_video_ids)


if __name__ == "__main__":
    split_data()
