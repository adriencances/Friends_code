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

        if type_of_interaction == 5:
            continue
        
        annotation = [episode_number, shot_id, person1, person2, type_of_interaction]
        annotations.append(annotation)

    return annotations


def compute_stats_by_character_pairs():
    characters = sorted(["joey", "chandler", "phoebe", "rachel", "monica", "ross"])
    character_pairs = []
    for i in range(6):
        for j in range(i + 1, 6):
            pair = (characters[i], characters[j])
            character_pairs.append(pair)
    number_names = ["co-occurences", "interactions", "physicals", "non-physicals"]
    numbers_by_pair = dict([
            (pair, dict([(name, 0) for name in number_names]))
        for pair in character_pairs])

    annotations = []
    for episode_number in range(1, 26):
        annotations += gather_annotations(episode_number)

    for annotation in annotations:
        episode_number, shot_id, person1, person2, type_of_interaction = annotation
        interacting = (type_of_interaction in [0, 1, 2, 3])
        physical = (type_of_interaction in [0, 1])
        non_physical = (type_of_interaction in [2, 3])
        current_pair = tuple(sorted((person1, person2)))
        if current_pair in character_pairs:
            numbers_by_pair[current_pair]["co-occurences"] += 1
            if interacting:
                numbers_by_pair[current_pair]["interactions"] += 1
            if physical:
                numbers_by_pair[current_pair]["physicals"] += 1
            if non_physical:
                numbers_by_pair[current_pair]["non-physicals"] += 1
    
    output_dir = "results/by_character_pair"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    complete_data_file = "{}/complete_data.csv".format(output_dir)
    with open(complete_data_file, "w") as f:
        f.write(",".join(["char1", "char2"] + number_names) + "\n")
        for pair in character_pairs:
            char1, char2 = pair
            f.write(",".join(map(str, [char1, char2, *list(numbers_by_pair[pair].values())])) + "\n")
    
    summarized_data_file = "{}/summarized_data.csv".format(output_dir)
    fig, axes= plt.subplots(3, 5)
    with open(summarized_data_file, "w") as f:
        f.write(",".join(["char1", "char2", "percentage of interactions", "percentage of physical-interactions"]) + "\n")
        for pair_index, pair in enumerate(character_pairs):
            char1, char2 = pair
            pair_stats = numbers_by_pair[pair]
            interaction_percentage = int(pair_stats["interactions"] / pair_stats["co-occurences"] * 100)
            physical_percentage = int(pair_stats["physicals"] / pair_stats["interactions"] * 100)
            f.write(",".join(map(str, [char1, char2, interaction_percentage, physical_percentage])) + "\n")

            # Pie chart
            class_names = ["no_interactions", "non-physical", "physical"]
            values = [pair_stats["physicals"], pair_stats["non-physicals"], pair_stats["co-occurences"] - pair_stats["interactions"]]
            ax = axes.flatten()[pair_index]
            ax.pie(values, startangle=90)
            ax.set_title("\n".join(pair))

            if pair_index == 14:
                fig.legend(labels=["physical", "non-physical", "no interaction"], loc='lower center', ncol=3)


            # fig1, ax1 = plt.subplots()
            # patches, texts = ax1.pie(detailed_data[:-1], startangle=90)
            # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            # plt.legend(patches, class_names, loc="best")
            # plt.title("Type of interactions in {}-character shots".format(key))
    fig.suptitle("Type of interactions by character pair", fontsize=14, y=0.12)
    plt.savefig("{}/piechart_character_pairs.pdf".format(output_dir))





if __name__ == "__main__":
    compute_stats_by_character_pairs()
