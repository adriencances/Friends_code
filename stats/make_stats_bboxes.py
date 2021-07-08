import os
from random import sample, seed
import sys
import glob
import subprocess
from pathlib import Path
import pickle
from numpy.lib.function_base import average
import tqdm
import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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


def get_temporal_intersection(track1, track2):
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


def get_overlap_over_whole_frame_values(annotation, info_by_shot):
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
            inter = get_temporal_intersection(track1, track2)
            if inter > max_inter:
                max_inter = inter
                best_ids = id1, id2
    
    id1, id2 = best_ids

    track1 = tracks[id1]
    track2 = tracks[id2]

    if not is_continuous(track1) or not is_continuous(track2):
        return None

    # Compute temporal intersection
    b1, e1 = tuple(map(int, track1[[0, -1], 0]))
    b2, e2 = tuple(map(int, track2[[0, -1], 0]))

    b = max(b1, b2)
    e = min(e1, e2)

    # If temporal intersection is too small, there can be no positive pairs
    temporal_intersection = max(0, e - b + 1)
    if temporal_intersection < SEGMENT_LENGTH:
        return None

    # Compute IoU vector to identify positive pairs
    tube1 = track1[b - b1:(e + 1) - b1, 1:5]
    tube2 = track2[b - b2:(e + 1) - b2, 1:5]

    overlap = overlap2d(tube2, tube1)
    whole_frame = 1280*720
    overlap_over_whole_frame = overlap / whole_frame

    return overlap_over_whole_frame


def compute_trajectories_and_averages_over_time():
    all_tracks = gather_tracks()
    averages_over_time = dict([("no interaction", []), ("non-physical", []), ("physical", [])])
    trajectories = dict([("no interaction", []), ("non-physical", []), ("physical", [])])
    for episode_number in tqdm.tqdm(range(1, 26)):
        annotations = gather_annotations(episode_number)
        info_by_shot = order_tracks(all_tracks, episode_number)
        for annotation in annotations:
            type_of_interaction = annotation[-1]
            if type_of_interaction == 5:
                continue
            values = get_overlap_over_whole_frame_values(annotation, info_by_shot)
            if values is None:
                continue
            average_value = np.mean(values)
            if type_of_interaction in [0, 1]:
                cat = "physical"
                trajectories[cat].append(values)
                averages_over_time[cat].append(average_value)
            elif type_of_interaction in [2, 3]:
                cat = "non-physical"
                trajectories[cat].append(values)
                averages_over_time[cat].append(average_value)
            else:
                cat = "no interaction"
                trajectories[cat].append(values)
                averages_over_time[cat].append(average_value)

    mean_trajectory = {}
    for cat in trajectories:
        max_length = max([len(traj) for traj in trajectories[cat]])
        for i in range(len(trajectories[cat])):
            traj = trajectories[cat][i]
            length = len(traj)
            interpol_func = interp1d(np.linspace(0, 1, length), traj, kind="previous")
            trajectories[cat][i] = interpol_func(np.linspace(0, 1, max_length))
        mean_trajectory[cat] = np.mean(np.array(trajectories[cat]), axis=0)

    return mean_trajectory, averages_over_time


def make_histograms():
    mean_trajectory, averages_over_time = compute_trajectories_and_averages_over_time()

    output_dir = "results/bboxes_stats"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes= plt.subplots(1, 3)
    max_value = max([max(averages_over_time[cat]) for cat in averages_over_time])
    colors = ["royalblue", "orangered", "olivedrab"]
    for index, ax in enumerate(axes.flatten()):
        cat = list(averages_over_time.keys())[index]
        values = averages_over_time[cat]

        ax.hist(values, density=True, bins=20, color=colors[index])
        ax.set_xlim((0, max_value))
        ax.set_ylim((0, 50))
        if index == 0:
            ax.set_ylabel("Frequency")
        ax.set_title(cat)

    plt.subplots_adjust(top=0.7)
    fig.suptitle("Histograms of bboxes intersection over\nwhole frame averaged over time for each pair", fontsize=14, y=0.9)
    plt.savefig("{}/hist_averages_over_time.pdf".format(output_dir))


    fig, axes= plt.subplots(3, 1)
    fig.tight_layout(pad=3)
    max_value = max([max(mean_trajectory[cat]) for cat in mean_trajectory])
    colors = ["royalblue", "orangered", "olivedrab"]
    for index, ax in enumerate(axes.flatten()):
        cat = list(averages_over_time.keys())[index]
        traj = mean_trajectory[cat]

        ax.plot(np.linspace(0, 1, len(traj)), traj, color=colors[index])
        ax.set_ylim((0, 1.2*max_value))
        ax.set_title(cat)

    plt.subplots_adjust(top=0.7)
    fig.suptitle("Evolution of bboxes intersection over whole frame,\naveraged over all pairs", fontsize=14, y=0.9)
    plt.savefig("{}/hist_mean_trajectories.pdf".format(output_dir))


if __name__ == "__main__":
    make_histograms()
