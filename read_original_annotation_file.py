import sys


def read_original_annotation_file(annotation_file):
    with open(annotation_file, "r") as f:
        nb_frames = int(f.readline().strip().split()[-1])

        boxes_by_frame = [[] for i in range(nb_frames)]
        frame_id = -1
        for line in f:
            if line[:6] == "#frame":
                frame_id += 1
            else:
                x1, y1, side = tuple(map(int, line.split()[1:4]))
                x2, y2 = x1 + side, y1 + side
                interacting = (line.split()[4] != "no_interaction")

                box_info = [(x1, y1, x2, y2), interacting]
                boxes_by_frame[frame_id].append(box_info)

    return boxes_by_frame


if __name__ == "__main__":
    annotation_file = sys.argv[1]
    boxes_by_frame = read_original_annotation_file(annotation_file)
