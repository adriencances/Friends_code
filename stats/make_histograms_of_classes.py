from os import get_terminal_size
import sys
import glob
import subprocess
from pathlib import Path
import pickle
import tqdm
import os.path

import numpy as np
import matplotlib.pyplot as plt

from textwrap import fill
from pandas_ods_reader import read_ods


main_dir = "/media/hdd/adrien/Friends"
annotations_dir = main_dir + "/annotations/filled"


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
        
        annotation = [episode_number, shot_id, person1, person2, type_of_interaction]
        annotations.append(annotation)

    return annotations


def make_histogram_of_classes():
    annotations = []
    for episode_number in range(1, 26):
        annotations += gather_annotations(episode_number)

    nb_annotations_by_class = [0 for class_id in range(6)]
    for annotation in annotations:
        class_id = annotation[4]
        nb_annotations_by_class[class_id] += 1

    class_names = ["Full physical", "Partial physical", "Full non-physical", "Partial non-physical", "No interaction", "Undefined"]
    class_names = [fill(name, 12) for name in class_names]
    
    # Bar plot
    fig = plt.figure(figsize = (10, 5))
    
    plt.bar(class_names, nb_annotations_by_class, color ='maroon', width = 0.4)
    
    plt.xlabel("Interaction class")
    plt.ylabel("Number of occurences")
    plt.title("Number of occurences by interaction class")
    plt.savefig("results/histogram_of_classes.pdf")


def get_stats_by_nb_of_characters():
    annotations = []
    for episode_number in range(1, 26):
        annotations += gather_annotations(episode_number)
    
    stats_by_nb_of_characters = {}
    detailed_stats_by_nb_of_characters = {}

    unique_shot_id = annotations[0][0]*1000 + annotations[0][1]
    characters = []
    nb_interacting_pairs = 0
    nb_of_each_class = [0 for c in range(6)]
    for annotation in annotations:
        episode_number, shot_id, person1, person2, type_of_interaction = annotation

        current_unique_shot_id = episode_number*1000 + shot_id
        if current_unique_shot_id != unique_shot_id:
            # Write stats of previous shot
            nb_characters = len(characters)
            if nb_characters not in stats_by_nb_of_characters:
                stats_by_nb_of_characters[nb_characters] = []
                detailed_stats_by_nb_of_characters[nb_characters] = [0 for c in range(6)]
            stats_by_nb_of_characters[nb_characters].append(nb_interacting_pairs)
            for c in range(6):
                detailed_stats_by_nb_of_characters[nb_characters][c] += nb_of_each_class[c]

            # Initialize stats of current shot
            unique_shot_id = current_unique_shot_id
            characters = []
            nb_interacting_pairs = 0
            nb_of_each_class = [0 for c in range(6)]

        # Update stats of current shot
        for person in [person1, person2]:
            if person not in characters:
                characters.append(person)
        if type_of_interaction not in [4, 5]:
            nb_interacting_pairs += 1
        
        nb_of_each_class[type_of_interaction] += 1

    # Write stats of last shot
    nb_characters = len(characters)
    if nb_characters not in stats_by_nb_of_characters:
        stats_by_nb_of_characters[nb_characters] = []
        detailed_stats_by_nb_of_characters[nb_characters] = [0 for c in range(6)]
    stats_by_nb_of_characters[nb_characters].append(nb_interacting_pairs)
    for c in range(6):
        detailed_stats_by_nb_of_characters[nb_characters][c] += nb_of_each_class[c]

    return stats_by_nb_of_characters, detailed_stats_by_nb_of_characters


def make_histogram_of_stats_by_nb_of_characters():
    output_dir = "results/k-character_shots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats, detailed_stats = get_stats_by_nb_of_characters()
    keys_to_consider = [2, 3, 4, 5, 6]
    for key in keys_to_consider:
        data = stats[key]
        detailed_data = detailed_stats[key]

        max_nb_interacting_pairs = key * (key - 1) // 2
        names = list(map(str, range(max_nb_interacting_pairs + 1)))
        values = [data.count(i) for i in range(max_nb_interacting_pairs + 1)]

        # Bar plot
        fig = plt.figure(figsize = (10, 5))
    
        plt.bar(names, values, color ='maroon', width = 0.4)
        
        plt.xlabel("Number of interacting pairs")
        plt.ylabel("Number of occurences")
        plt.title("Number of interacting pairs in {}-character shots".format(key))
        plt.savefig("{}/hist_interactions_{}-char_shots.pdf".format(output_dir, key))

        # Pie chart
        class_names = ["Full physical", "Partial physical", "Full non-physical", "Partial non-physical", "No interaction"]

        fig1, ax1 = plt.subplots()
        patches, texts = ax1.pie(detailed_data[:-1], startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.legend(patches, class_names, loc="best")
        plt.title("Type of interactions in {}-character shots".format(key))
        plt.savefig("{}/piechart_interactions_{}-char_shots.pdf".format(output_dir, key))


if __name__ == "__main__":
    make_histogram_of_classes()
    make_histogram_of_stats_by_nb_of_characters()
