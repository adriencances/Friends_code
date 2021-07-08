from os import get_terminal_size
from random import sample, seed
import sys
import glob
import subprocess
from pathlib import Path
import pickle
import tqdm
import os.path

import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import floor, ceil
from pandas_ods_reader import read_ods

sys.path.insert(0, "/home/acances/Code/Friends")
from make_spreadsheets import gather_tracks, order_tracks


SEGMENT_LENGTH = 16


main_dir = "/media/hdd/adrien/Friends"
shots_dir = main_dir + "/shots"
annotations_dir = main_dir + "/annotations/filled"
pairs_dir = main_dir + "/pairs16"

tracks_file = main_dir + "/tracks-features/Friends_final.pk"


def gather_annotations(episode_number):
    annotation_file = "{}/annotations_ep{:02d}_filled.ods".format(annotations_dir, episode_number)
    df = read_ods(annotation_file, 1)
    
    annotations = []
    for row in range(1, len(df)):
        entries = df.iloc[row].tolist()
        shot_id = int(entries[1])
        person1, person2 = entries[2:4]
        interactions = entries[4:]
        if 'x' not in interactions:
            continue
        type_of_interaction = interactions.index("x")

        # Ignore undefined annotation
        if type_of_interaction == 5:
            continue

        annotation = [episode_number, shot_id, person1, person2, type_of_interaction]
        annotations.append(annotation)

    return annotations


def temporal_intersection(track1, track2):
    """
    returns the size (in frames) of the temporal intersection of the two tracks
    """

    b1, e1 = track1[[0, -1], 0]
    b2, e2 = track2[[0, -1], 0]

    b = max(b1, b2)
    e = min(e1, e2)

    inter = max(0, e - b + 1)

    return inter


def is_continuous(track):
    b = int(track[0, 0])
    e = int(track[-1, 0])
    return e - b + 1 == len(track)


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


def get_pos_pairs_begin_indices(iou, IOU_THRESHOLD, FRAME_PROPORTION):
    """Returns the list of beginning indices of the positive segments,
    given the list of IoUs of the intersection of two tracks.
    Thus, the returned indices are relative to intersection of the two tracks."""
    is_above = iou > IOU_THRESHOLD
    N = len(is_above)
    begin_indices = []
    # Segments are chosen to be pairwise disjoint
    j = 0
    while True:
        if j + SEGMENT_LENGTH > N: break
        # Check whether or not there are enough frames with IoU above the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) / SEGMENT_LENGTH >= FRAME_PROPORTION:
            begin_indices.append(j)
            j += SEGMENT_LENGTH
        else:
            j += 1
    return begin_indices


def get_medneg_pairs_begin_indices(iou, IOU_THRESHOLD, FRAME_PROPORTION):
    """Returns the list of beginning indices of the medium_negative segments,
    given the list of IoUs of the intersection of two tracks.
    Thus, the returned indices are relative to intersection of the two tracks."""
    is_above = iou > IOU_THRESHOLD
    N = len(is_above)
    begin_indices = []
    # Segments are chosen to be pairwise disjoint
    j = 0
    while True:
        if j + SEGMENT_LENGTH > N: break
        # Check whether or not all frames have IoU below the threshold
        if np.sum(is_above[j:j+SEGMENT_LENGTH]) == 0:
            begin_indices.append(j)
            j += SEGMENT_LENGTH
        else:
            j += 1
    return begin_indices


def get_pos_and_medneg_pairs_begin_frames_for_tracks(track1, track2, IOU_THRESHOLD, FRAME_PROPORTION):
    """Returns the list of begin_frame indices of positive pairs and of medium negative pairs.
    The indices are relative to the shot to which the two tracks belong."""
    # tr1 : [[frame_idx, x1, y1, x2, y2, score], ...]

    # Compute temporal intersection
    b1, e1 = tuple(map(int, track1[[0, -1], 0]))
    b2, e2 = tuple(map(int, track2[[0, -1], 0]))

    b = max(b1, b2)
    e = min(e1, e2)

    # If temporal intersection is too small, there can be no positive pairs
    temporal_intersection = max(0, e - b + 1)
    if temporal_intersection < SEGMENT_LENGTH:
        return [], []

    # Compute IoU vector to identify positive pairs
    tube1 = track1[b - b1:(e + 1) - b1, 1:5]
    tube2 = track2[b - b2:(e + 1) - b2, 1:5]

    iou = iou2d(tube1, tube2)
    pos_begin_indices = get_pos_pairs_begin_indices(iou, IOU_THRESHOLD, FRAME_PROPORTION)
    medneg_begin_indices = get_medneg_pairs_begin_indices(iou, IOU_THRESHOLD, FRAME_PROPORTION)

    return pos_begin_indices, medneg_begin_indices


def apply_criterion(annotation, stats, info_by_shot, IOU_THRESHOLD, FRAME_PROPORTION):
    episode_number, shot_id, person1, person2, type_of_interaction = annotation

    tracks_by_shot, track_ids_by_shot, characters_by_shot = info_by_shot

    tracks = tracks_by_shot[shot_id]
    characters = characters_by_shot[shot_id]
    track_ids = track_ids_by_shot[shot_id]

    track_ids_1 = [i for i in range(len(tracks)) if characters[i] == person1]
    track_ids_2 = [i for i in range(len(tracks)) if characters[i] == person2]

    max_inter = -1
    best_ids = None
    for id1 in track_ids_1:
        for id2 in track_ids_2:
            track1 = tracks[id1]
            track2 = tracks[id2]
            inter = temporal_intersection(track1, track2)
            if inter > max_inter:
                max_inter = inter
                best_ids = id1, id2
    
    id1, id2 = best_ids

    track1 = tracks[id1]
    track2 = tracks[id2]

    if not is_continuous(track1) or not is_continuous(track2):
        return []

    # APPLY CRITERION
    pos_begin_frames, medneg_begin_frames = get_pos_and_medneg_pairs_begin_frames_for_tracks(
        track1, track2, IOU_THRESHOLD, FRAME_PROPORTION
    )

    criterion_gives_positives = (len(pos_begin_frames) > 0)
    criterion_gives_negatives = (len(medneg_begin_frames))

    full_interaction = (type_of_interaction in [0, 2])
    partial_interaction = (type_of_interaction in [1, 3])

    if full_interaction:
        if criterion_gives_positives:
            stats["TP"] += 1
        else:
            stats["MP"] += 1
        if criterion_gives_negatives:
            stats["FN"] += 1
    elif partial_interaction:
        if criterion_gives_positives:
            stats["TP"] += 1
        else:
            stats["MP"] += 1
        if criterion_gives_negatives:
            stats["TN"] += 1
        else:
            stats["MN"] += 1
    else:
        if criterion_gives_positives:
            stats["FP"] += 1
        if criterion_gives_negatives:
            stats["TN"] += 1
        else:
            stats["MN"] += 1


def update_stats(episode_number, stats, all_tracks, IOU_THRESHOLD, FRAME_PROPORTION):
    info_by_shot = order_tracks(all_tracks, episode_number)
    annotations = gather_annotations(episode_number)

    for annotation in annotations:
        apply_criterion(annotation, stats, info_by_shot, IOU_THRESHOLD, FRAME_PROPORTION)


def compute_stats(IOU_THRESHOLD, FRAME_PROPORTION, all_tracks=None, hide_progress_bar=False):
    if all_tracks is None:
        all_tracks = gather_tracks()
    stats = dict([(cat, 0) for cat in ["TP", "FP", "MP", "TN", "FN", "MN"]])
    for episode_number in tqdm.tqdm(range(1, 26), disable=hide_progress_bar):
        update_stats(episode_number, stats, all_tracks, IOU_THRESHOLD, FRAME_PROPORTION)
    return stats


def compute_stats_for_several_configs():
    output_file = "results/criterion_stats/criterion_stats_by_config.csv"
    Path("/".join(output_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(",".join(["IOU_THRESHOLD", "FRAME_PROPORTION", "TP", "FP", "MP", "TN", "FN", "MN"]) + "\n")

    all_tracks = gather_tracks()
    stats_by_config = {}
    IOU_THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    FRAME_PROPORTION_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    configurations = list(itertools.product(IOU_THRESHOLD_VALUES, FRAME_PROPORTION_VALUES))
    for config in tqdm.tqdm(configurations):
        stats_by_config[config] = compute_stats(*config, all_tracks, hide_progress_bar=True)
        with open(output_file, "a") as f:
            entries = list(config) + list(stats_by_config[config].values())
            f.write(",".join(map(str, entries)) + "\n")


def make_graphs_for_the_configs():
    output_file = "results/criterion_stats/criterion_stats_by_config.csv"
    stats = {}
    categories = ["TP", "FP", "MP", "TN", "FN", "MN"]
    pos_categories = categories[:3]
    neg_categories = categories[3:]
    with open(output_file, "r") as f:
        f.readline()
        for line in f:
            entries = line.strip().split(",")
            config = tuple(map(float, entries[:2]))
            result = list(map(int, entries[2:]))
            stats[config] = dict([(categories[i], result[i]) for i in range(len(categories))])
    
    IOU_THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    FRAME_PROPORTION_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    max_by_cat = dict([(cat, max([stats[k][cat] for k in stats])) for cat in categories])
    output_dir = "results/criterion_stats"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for cat in neg_categories:
        FRAME_PROPORTION = FRAME_PROPORTION_VALUES[0]
        values = [stats[IOU_THRESHOLD, FRAME_PROPORTION][cat] for IOU_THRESHOLD in IOU_THRESHOLD_VALUES]

        fig = plt.figure(figsize = (10, 5))
        plt.bar(IOU_THRESHOLD_VALUES, values, color ='orangered', width=0.05)
        plt.ylim(top=1.2*max_by_cat[cat])
        plt.xlabel("IoU threshold")
        plt.ylabel("Number of {}".format(cat))
        plt.title("Number {} for different IoU thresholds\n".format(cat))
        plt.savefig("{}/histogram_{}.pdf".format(output_dir, cat))
        plt.close()

    for FRAME_PROPORTION in FRAME_PROPORTION_VALUES:
        for cat in pos_categories:
            values = [stats[IOU_THRESHOLD, FRAME_PROPORTION][cat] for IOU_THRESHOLD in IOU_THRESHOLD_VALUES]

            fig = plt.figure(figsize = (10, 5))
            plt.bar(IOU_THRESHOLD_VALUES, values, color ='royalblue', width=0.05)
            plt.ylim(top=1.2*max_by_cat[cat])
            plt.xlabel("IoU threshold")
            plt.ylabel("Number of {}".format(cat))
            plt.title("Number {} for different IoU thresholds\nwhen frame proportion is {}".format(cat, FRAME_PROPORTION))
            plt.savefig("{}/histogram_{}_frame_proportion_{}.pdf".format(output_dir, cat, FRAME_PROPORTION))
            plt.close()


if __name__ == "__main__":
    make_graphs_for_the_configs()
