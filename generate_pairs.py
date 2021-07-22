from os import get_terminal_size
from random import sample, seed
import sys
import glob
import subprocess
from pathlib import Path
import pickle
import tqdm
import os.path

from math import floor, ceil
from pandas_ods_reader import read_ods

from make_spreadsheets import gather_shots, gather_tracks, generate_all_problematic_shot_dirs, order_tracks


main_dir = "/home/adrien/Data/Friends"
shots_dir = main_dir + "/shots"
annotations_dir = main_dir + "/annotations/filled"
pairs_dir = main_dir + "/pairs16"

tracks_file = main_dir + "/tracks-features/Friends_final.pk"


SEGMENT_LENGTH = 16

seed(0)


def gather_annotations(episode_number):
    annotation_file = "{}/annotations_ep{:02d}_filled.ods".format(annotations_dir, episode_number)
    df = read_ods(annotation_file, 1)
    
    pos_annotations = []
    neg_annotations = []
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
        
        # No interaction
        annotation = [episode_number, shot_id, person1, person2, type_of_interaction]
        if type_of_interaction == 4:
            neg_annotations.append(annotation)
        else:
            pos_annotations.append(annotation)

    return pos_annotations, neg_annotations


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


def sample_modulo_1frame(begin, end, segment_length=SEGMENT_LENGTH):
    step = ceil((end - begin + 1) / segment_length)
    rest = segment_length*step - (end - begin + 1)
    nb_shifts = step - rest
    lists_of_frames = []
    for shift in range(nb_shifts):
        lists_of_frames.append([begin + shift + k*step for k in range(segment_length)])
    return lists_of_frames


def sample_modulo_2frame(begin, end):
    lists_of_frames_1frame = sample_modulo_1frame(begin, end, segment_length=8)
    nb_2frame = len(lists_of_frames_1frame) - 1
    lists_of_frames_2frame = []
    for index in range(nb_2frame):
        frames_list = []
        for shift in range(index, index + 2):
            frames_list += lists_of_frames_1frame[shift]
        lists_of_frames_2frame.append(sorted(frames_list))
    return lists_of_frames_2frame


def sample_modulo_4frame(begin, end):
    lists_of_frames_1frame = sample_modulo_1frame(begin, end, segment_length=4)
    nb_4frame = len(lists_of_frames_1frame) - 3
    lists_of_frames_4frame = []
    for index in range(nb_4frame):
        frames_list = []
        for shift in range(index, index + 4):
            frames_list += lists_of_frames_1frame[shift]
        lists_of_frames_4frame.append(sorted(frames_list))
    return lists_of_frames_4frame


def sample_random(begin, end):
    if end - begin + 1 < 2 * SEGMENT_LENGTH:
        return []
    lists_of_frames = []
    for rep in range(10):
        lists_of_frames.append(sorted(sample(list(range(begin, end)), SEGMENT_LENGTH)))
    return lists_of_frames


def get_lists_of_frames(track1, track2, type_of_interaction):
    b1, e1 = track1[[0, -1], 0]
    b2, e2 = track2[[0, -1], 0]

    begin = int(max(b1, b2))
    end = int(min(e1, e2))

    if end - begin + 1 < SEGMENT_LENGTH:
        return []

    # No interaction
    if type_of_interaction == 4:
        begin_frames = list(range(begin, end - SEGMENT_LENGTH + 2, SEGMENT_LENGTH))
        lists_of_frames = [list(range(b, b + SEGMENT_LENGTH)) for b in begin_frames]
        lists_of_frames += sample_modulo_1frame(begin, end)
        lists_of_frames += sample_modulo_2frame(begin, end)
        lists_of_frames += sample_modulo_4frame(begin, end)
        # lists_of_frames += sample_random(begin, end)
    # Full interaction or no interaction
    elif type_of_interaction in [0, 2]:
        begin_frames = list(range(begin, end - SEGMENT_LENGTH + 2, SEGMENT_LENGTH))
        lists_of_frames = [list(range(b, b + SEGMENT_LENGTH)) for b in begin_frames]
        lists_of_frames += sample_modulo_1frame(begin, end)
        lists_of_frames += sample_modulo_2frame(begin, end)
        lists_of_frames += sample_modulo_4frame(begin, end)
    # Partial interactions
    elif type_of_interaction in [1, 3]:
        lists_of_frames = []
        lists_of_frames += sample_modulo_1frame(begin, end)
        lists_of_frames += sample_modulo_2frame(begin, end)
        lists_of_frames += sample_modulo_4frame(begin, end)
    else:
        return []
    
    for frames_list in lists_of_frames:
        assert len(frames_list) == SEGMENT_LENGTH

    return lists_of_frames


def is_continuous(track):
    b = int(track[0, 0])
    e = int(track[-1, 0])
    return e - b + 1 == len(track)


def get_pair_lists_of_frames_hard_negative(track1, track2, type_of_interaction):
    if type_of_interaction == 4:
        return []

    b1, e1 = track1[[0, -1], 0]
    b2, e2 = track2[[0, -1], 0]

    begin = int(max(b1, b2))
    end = int(min(e1, e2))

    if end - begin + 1 < 2*SEGMENT_LENGTH + 10:
        return []

    frame_indices1 = list(range(begin, begin + 16))
    frame_indices2 = list(range(end - 15, end + 1))

    pair_lists_of_frames = [(frame_indices1, frame_indices2), (frame_indices2, frame_indices1)]

    return pair_lists_of_frames


def generate_pairs_for_annotation(annotation, info_by_shot, label):
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
        return [], []

    lists_of_frames = get_lists_of_frames(track1, track2, type_of_interaction)

    track_id1 = track_ids[id1]
    track_id2 = track_ids[id2]
    pairs = []
    pairs += [[episode_number, track_id1, track_id2, *frames_list, *frames_list, label] for frames_list in lists_of_frames]

    pair_lists_of_frames = get_pair_lists_of_frames_hard_negative(track1, track2, type_of_interaction)
    hard_neg_pairs = [[episode_number, track_id1, track_id2, *frames_list1, *frames_list2, 0] for frames_list1, frames_list2 in pair_lists_of_frames]

    return pairs, hard_neg_pairs


def generate_pairs_for_episode(episode_number, all_tracks=None):
    if all_tracks is None:
        all_tracks = gather_tracks()
    info_by_shot = order_tracks(all_tracks, episode_number)
    pos_annotations, neg_annotations = gather_annotations(episode_number)

    pos_pairs = []
    hard_neg_pairs = []
    for annotation in pos_annotations:
        pairs, hard_pairs = generate_pairs_for_annotation(annotation, info_by_shot, label=1)
        pos_pairs += pairs
        hard_neg_pairs += hard_pairs

    neg_pairs = []
    for annotation in neg_annotations:
        pairs, hard_pairs = generate_pairs_for_annotation(annotation, info_by_shot, label=0)
        neg_pairs += pairs
        hard_neg_pairs += hard_pairs
    
    # neg_pairs += hard_neg_pairs

    pos_subdir = "{}/positive/".format(pairs_dir, episode_number)
    Path(pos_subdir).mkdir(parents=True, exist_ok=True)
    neg_subdir = "{}/negative/".format(pairs_dir, episode_number)
    Path(neg_subdir).mkdir(parents=True, exist_ok=True)

    pos_pairs_file = "{}/pairs_episode{:02d}.csv".format(pos_subdir, episode_number)
    neg_pairs_file = "{}/pairs_episode{:02d}.csv".format(neg_subdir, episode_number)

    with open(pos_pairs_file, "w") as f:
        for pair in pos_pairs:
            f.write(",".join(map(str, pair)) + "\n")
    
    with open(neg_pairs_file, "w") as f:
        for pair in neg_pairs:
            f.write(",".join(map(str, pair)) + "\n")
    
    return len(hard_neg_pairs)


def generate_all_pairs():
    all_tracks = gather_tracks()
    nb_hard_negs = 0
    for episode_number in tqdm.tqdm(range(1, 26)):
        annotation_file = "{}/annotations_ep{:02d}_filled.ods".format(annotations_dir, episode_number)
        if not os.path.isfile(annotation_file):
            continue
        nb_hard_negs += generate_pairs_for_episode(episode_number, all_tracks)
    print("nb hard negatives: {}".format(nb_hard_negs))


if __name__ == "__main__":
    generate_all_pairs()
