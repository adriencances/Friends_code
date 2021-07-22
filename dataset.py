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
import torchvision

from dataset_aux import FrameProcessor


class FriendsPairs(data.Dataset):
    def __init__(self, phase="train", eight_frames=False, augmented=False, nb_positives=None, seed=0):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.eight_frames = eight_frames
        self.augmented = augmented
        self.frames_dir = "/home/acances/Data/Friends/frames"
        self.pairs_dir = "/home/acances/Data/Friends/pairs16"
        self.tracks_file = "/home/acances/Data/Friends/tracks-features/Friends_final.pk"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.tracks_file)

        random.seed(seed)
        self.nb_positives = nb_positives
        self.gather_episode_numbers()
        self.gather_positive_pairs()
        self.gather_negative_pairs()
        self.create_data()
        self.init_augmentation_transforms()
    
    def init_augmentation_transforms(self):
        self.color_jitter = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.05)
        # Remark: because of the format of the data, this function will operate a horizontal flip
        self.horizontal_flip = torchvision.transforms.RandomVerticalFlip(p=0.5)
    
    def augmented_tensor(self, tensor):
        # Switch format from (C, T, W, H) to (T, C, W, H) and divide by 255 to apply image transforms
        u = tensor.permute(1, 0, 2, 3) / 255
        u = self.color_jitter(u)
        u = self.horizontal_flip(u)
        tensor = u.permute(1, 0, 2, 3) * 255
        return tensor
    
    def gather_episode_numbers(self):
        self.episode_numbers = []
        split_file = "/home/acances/Data/Friends/split/{}.txt".format(self.phase)
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
                    frame_indices1 = list(map(int, entries[3:19]))
                    frame_indices2 = list(map(int, entries[19:35]))
                    label = int(entries[-1])
                    pair = [episode_number, track_id1, track_id2, frame_indices1, frame_indices2, label]
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
                    frame_indices1 = list(map(int, entries[3:19]))
                    frame_indices2 = list(map(int, entries[19:35]))
                    label = int(entries[-1])
                    pair = [episode_number, track_id1, track_id2, frame_indices1, frame_indices2, label]
                    self.negative_pairs.append(pair)
        # nb_negatives = min(3 * self.nb_positives, len(self.negative_pairs))
        # self.negative_pairs = random.sample(self.negative_pairs, nb_negatives)

    def create_data(self):
        data_size = min(4 * len(self.positive_pairs), 4 * len(self.negative_pairs) // 3)
        nb_positives = data_size // 4
        nb_negatives = 3 * data_size // 4

        self.positive_pairs = random.sample(self.positive_pairs, nb_positives)
        self.negative_pairs = random.sample(self.negative_pairs, nb_negatives)

        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        random.shuffle(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        assert index < len(self.data)
        
        pair = self.data[index]

        episode_number, track_id1, track_id2, frame_indices1, frame_indices2, label = pair

        tensor1 = self.frame_processor.processed_frames(episode_number, track_id1, frame_indices1)
        tensor2 = self.frame_processor.processed_frames(episode_number, track_id2, frame_indices2)

        # Augment data
        if self.augmented:
            tensor12 = torch.cat((tensor1, tensor2), dim=1)
            tensor12 = self.augmented_tensor(tensor12)
            tensor1 = tensor12[:,:16]
            tensor2 = tensor12[:,16:]

        # Downsample by two to get 8-frame tensors
        if self.eight_frames:
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
    dataset = FriendsPairs("train", augmented=True)
    print("nb positives: \t", len(dataset.positive_pairs))
    print("nb negatives: \t", len(dataset.negative_pairs))

    # for i in tqdm.tqdm(range(10)):
    #     t1, t2, lab = dataset[0]
    #     dataset.print_tensors(t1[:,:1], t2[:,:1], "augmented_examples/pair_0_version_{}".format(i))

    # for index in tqdm.tqdm(range(len(dataset))):
    #     content = dataset[index]

    # index = int(sys.argv[1])
    # tensor1, tensor2, label = dataset[index]
    # print(label)
    # dataset.print_tensors(tensor1, tensor2, subdir="visualisation")
