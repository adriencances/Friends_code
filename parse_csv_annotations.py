import sys
import glob
from pathlib import Path
import tqdm
import os.path

from math import floor, ceil

main_dir = "/media/hdd/adrien/Friends"
annotations_dir = main_dir + "/annotations/filled_csv"

labels = (["full physical", "partial physical",
            "full non-physical", "partial non-physical",
            "no interaction", "undefined"])


def gather_annotations(episode_number):
    annotation_file = "{}/annotations_ep{:02d}_filled.csv".format(annotations_dir, episode_number)
    annotations = dict([(label, []) for label in labels])
    with open(annotation_file, "r") as f:
        f.readline()
        for line in f:
            entries = line.strip().split(",")
            shot_id = int(entries[1])
            person1, person2 = entries[2:4]
            interactions = entries[4:]
            if 'x' not in interactions:
                continue
            type_of_interaction = interactions.index("x")
            annotation = [episode_number, shot_id, person1, person2]
            annotations[labels[type_of_interaction]].append(annotation)

    return annotations


def gather_all_annotations():
    all_annotations = dict([(label, []) for label in labels])
    for episode_number in range(1, 26):
        annotations = gather_annotations(episode_number)
        for k in annotations:
            all_annotations[k] += annotations[k]

    return all_annotations
