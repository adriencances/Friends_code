import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle
from pathlib import Path

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor


class FriendsPairs(data.Dataset):
    def __init__(self, phase="train", nb_positives=None, seed=0):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.frames_dir = "/media/hdd/adrien/Friends/frames"
        self.pairs_dir = "/media/hdd/adrien/Friends/pairs16"
        self.tracks_file = "/media/hdd/adrien/Friends/tracks-features/Friends_final.pk"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.tracks_file)

        random.seed(seed)
        self.nb_positives = nb_positives
        self.gather_episode_numbers()
        self.gather_positive_pairs()
        self.gather_negative_pairs()
        self.create_data()
    
    def gather_episode_numbers(self):
        self.episode_numbers = []
        split_file = "/media/hdd/adrien/Friends/split/{}.txt".format(self.phase)
        with open(split_file, "r") as f:
            for line in f:
                episode_number = int(line.strip())
                self.episode_numbers.append(episode_number)

    def gather_positive_pairs(self):
        # print("Gathering positive pairs")
        self.positive_pairs = []
        pairs_files = glob.glob("{}/positive/*".format(self.pairs_dir))
        for file in pairs_files:
            episode_number = int(file.split("/")[-1].split(".")[0][-2:])
            if episode_number not in self.episode_numbers:
                continue
            with open(file, "r") as f:
                for line in f:
                    entries = line.strip().split(",")
                    episode_number = int(entries[0])
                    track_id1 = int(entries[1])
                    track_id2 = int(entries[2])
                    frame_indices = list(map(int, entries[3:19]))
                    label = int(entries[-1])
                    pair = [episode_number, track_id1, track_id2, frame_indices, label]
                    self.positive_pairs.append(pair)
        if self.nb_positives is None:
            self.nb_positives = len(self.positive_pairs)
        self.positive_pairs = random.sample(self.positive_pairs, self.nb_positives)

    def gather_negative_pairs(self):
        # print("Gathering negative pairs")
        self.negative_pairs = []
        pairs_files = glob.glob("{}/negative/*".format(self.pairs_dir))
        for file in pairs_files:
            episode_number = int(file.split("/")[-1].split(".")[0][-2:])
            if episode_number not in self.episode_numbers:
                continue
            with open(file, "r") as f:
                for line in f:
                    entries = line.strip().split(",")
                    episode_number = int(entries[0])
                    track_id1 = int(entries[1])
                    track_id2 = int(entries[2])
                    frame_indices = list(map(int, entries[3:19]))
                    label = int(entries[-1])
                    pair = [episode_number, track_id1, track_id2, frame_indices, label]
                    self.negative_pairs.append(pair)
        nb_negatives = 3 * self.nb_positives
        self.negative_pairs = random.sample(self.negative_pairs, nb_negatives)
    
    def create_data(self):
        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        random.shuffle(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        assert index < len(self.data)
        
        pair = self.data[index]

        episode_number, track_id1, track_id2, frame_indices, label = pair

        tensor1 = self.frame_processor.processed_frames(episode_number, track_id1, frame_indices)
        tensor2 = self.frame_processor.processed_frames(episode_number, track_id2, frame_indices)

        # Downsample by two to get 8-frame tensors
        tensor1 = tensor1[:,::2]
        tensor2 = tensor2[:,::2]
        
        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)
    
    def print_tensors(self, tensor1, tensor2, subdir):
        Path(subdir).mkdir(parents=True, exist_ok=True)

        for i in range(tensor1.shape[1]):
            filename1 = "{}/tensor1_frame_{}.jpg".format(subdir, i + 1)
            frame1 = tensor1[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename1, frame1)

            filename2 = "{}/tensor2_frame_{}.jpg".format(subdir, i + 1)
            frame2 = tensor2[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename2, frame2)


if __name__ == "__main__":
    dataset = FriendsPairs("train")

    for index in tqdm.tqdm(range(len(dataset))):
        content = dataset[index]

    # index = int(sys.argv[1])
    # tensor1, tensor2, label = dataset[index]
    # print(label)
    # dataset.print_tensors(tensor1, tensor2, subdir="visualisation")
