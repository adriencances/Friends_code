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


def gather_annotations(episode_number, skip_undefined=True):
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
        if skip_undefined and type_of_interaction == 5:
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
    if len(track) == 0:
        return True
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


def get_nb_characters_by_shot(episode_number):
    annotations = gather_annotations(episode_number, skip_undefined=False)

    nb_characters_by_shot = {}

    previous_shot_id = annotations[0][1]
    characters = []
    for annotation in annotations:
        episode_number, shot_id, person1, person2, type_of_interaction = annotation

        if shot_id != previous_shot_id:
            # Write info of previous shot
            nb_characters = len(characters)
            nb_characters_by_shot[previous_shot_id] = nb_characters

            # Initialize stats of current shot
            previous_shot_id = shot_id
            characters = []

        # Update stats of current shot
        for person in [person1, person2]:
            if person not in characters:
                characters.append(person)

    # Write stats of last shot
    nb_characters = len(characters)
    nb_characters_by_shot[previous_shot_id] = nb_characters

    return nb_characters_by_shot


def get_iou_values(annotation, info_by_shot):
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

    iou = iou2d(tube2, tube1)

    return iou


def compute_trajectories_and_averages_over_time(nb_characters=None):
    all_tracks = gather_tracks()
    averages_over_time = dict([("no interaction", []), ("non-physical", []), ("physical", [])])
    trajectories = dict([("no interaction", []), ("non-physical", []), ("physical", [])])
    for episode_number in tqdm.tqdm(range(1, 26)):
        annotations = gather_annotations(episode_number)
        info_by_shot = order_tracks(all_tracks, episode_number)
        nb_characters_by_shot = get_nb_characters_by_shot(episode_number)
        for annotation in annotations:
            shot_id = annotation[1]
            if nb_characters is not None:
                if nb_characters_by_shot[shot_id] != nb_characters:
                    continue
                assert nb_characters_by_shot[shot_id] == nb_characters

            type_of_interaction = annotation[-1]
            if type_of_interaction == 5:
                continue
            values = get_iou_values(annotation, info_by_shot)
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


def make_histograms_iou(nb_characters=None):
    mean_trajectory, averages_over_time = compute_trajectories_and_averages_over_time(nb_characters)

    output_dir = "results/bboxes_stats"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes= plt.subplots(1, 3)
    fig.tight_layout()
    max_value = max([max(averages_over_time[cat]) for cat in averages_over_time])
    colors = ["royalblue", "orangered", "olivedrab"]
    for index, ax in enumerate(axes.flatten()):
        cat = list(averages_over_time.keys())[index]
        values = averages_over_time[cat]

        ax.hist(values, density=False, bins=20, color=colors[index])
        ax.set_xlim((0, max_value))
        if nb_characters is None:
            ax.set_ylim((0, 2000))
        else:
            ax.set_ylim((0, 750))
        ax.set_title(cat)

    plt.subplots_adjust(top=0.7)
    title = "Histograms of IoU averaged over time for each pair"
    if nb_characters is not None:
        title += "\nfor {}-character shots".format(nb_characters)
    fig.suptitle(title, fontsize=14, y=0.9)

    file_name = "hist_iou_averages_over_time"
    if nb_characters is not None:
        file_name += "_{}-char".format(nb_characters)
    plt.savefig("{}/{}.pdf".format(output_dir, file_name))
    plt.savefig("{}/{}.png".format(output_dir, file_name))


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
    title = "Evolution of IoU,\naveraged over all pairs"
    if nb_characters is not None:
        title += "\nfor {}-character shots".format(nb_characters)
    fig.suptitle(title, fontsize=14, y=0.9)
    
    file_name = "hist_iou_mean_trajectories"
    if nb_characters is not None:
        file_name += "_{}-char".format(nb_characters)
    plt.savefig("{}/{}.pdf".format(output_dir, file_name))
    plt.savefig("{}/{}.png".format(output_dir, file_name))


def get_average_head_over_body_value(track_id, episode_number, all_tracks):
    body_track = all_tracks["episode{:02d}".format(episode_number)]["body"][track_id]
    head_track = all_tracks["episode{:02d}".format(episode_number)]["face"][track_id]

    if not is_continuous(body_track) or not is_continuous(head_track):
        return None

    # Compute temporal intersection
    b1, e1 = tuple(map(int, body_track[[0, -1], 0]))
    b2, e2 = tuple(map(int, head_track[[0, -1], 0]))

    b = max(b1, b2)
    e = min(e1, e2)

    # Compute IoU vector to identify positive pairs
    tube1 = body_track[b - b1:(e + 1) - b1, 1:5]
    tube2 = head_track[b - b2:(e + 1) - b2, 1:5]

    if len(tube1) == 0 or len(tube2) == 0 or tube1.min() == 0:
        return None

    ratios = tube2 / tube1
    average_ratio = ratios.mean()

    return average_ratio


def get_average_head_over_body_ratio_values(annotation, info_by_shot, all_tracks):
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

    track_id_1 = track_ids[id1]
    track_id_2 = track_ids[id2]

    if not is_continuous(track1) or not is_continuous(track2):
        return []
    
    values = []

    if track_id_1 <= 9999:
        value = get_average_head_over_body_value(track_id_1, episode_number, all_tracks)
        if value is not None:
            values.append(value)
    
    if track_id_2 <= 9999:
        value = get_average_head_over_body_value(track_id_2, episode_number, all_tracks)
        if value is not None:
            values.append(value)

    return values


def compute_average_head_over_body_ratios():
    all_tracks = gather_tracks()
    values_by_type = dict([("no interaction", []), ("non-physical", []), ("physical", [])])
    for episode_number in tqdm.tqdm(range(1, 26)):
        annotations = gather_annotations(episode_number)
        info_by_shot = order_tracks(all_tracks, episode_number)
        for annotation in annotations:
            type_of_interaction = annotation[-1]
            if type_of_interaction == 5:
                continue
            values = get_average_head_over_body_ratio_values(annotation, info_by_shot, all_tracks)
            if type_of_interaction in [0, 1]:
                cat = "physical"
                values_by_type[cat] += values
            elif type_of_interaction in [2, 3]:
                cat = "non-physical"
                values_by_type[cat] += values
            else:
                cat = "no interaction"
                values_by_type[cat] += values

    return values_by_type


def make_histograms_head_over_body():
    values_by_type = compute_average_head_over_body_ratios()


    output_dir = "results/bboxes_stats"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes= plt.subplots(1, 3)
    max_value = max([max(values_by_type[cat]) for cat in values_by_type])
    colors = ["royalblue", "orangered", "olivedrab"]
    for index, ax in enumerate(axes.flatten()):
        cat = list(values_by_type.keys())[index]
        values = values_by_type[cat]

        ax.hist(values, density=True, bins=20, color=colors[index])
        ax.set_xlim((0, max_value))
        ax.set_ylim((0, 50))
        if index == 0:
            ax.set_ylabel("Frequency")
        ax.set_title(cat)

    plt.subplots_adjust(top=0.7)
    fig.suptitle("Histograms of head_over_body ratio averaged over time for each character", fontsize=14, y=0.9)
    plt.savefig("{}/hist_head_over_body.pdf".format(output_dir))
    plt.savefig("{}/hist_head_over_body.png".format(output_dir))


if __name__ == "__main__":
    make_histograms_iou()
    make_histograms_iou(nb_characters=2)
    make_histograms_iou(nb_characters=3)
    make_histograms_head_over_body()
    # tube1, tube2 = get_average_head_over_body_value(track_id_2, episode_number, all_tracks)
