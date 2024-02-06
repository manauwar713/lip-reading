import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
import dlib
import torch
import gdown

def load_video(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[190:236, 80:220]
        frame = cv2.resize(frame, (100, 50))

        # Convert to NumPy and append to frames
        frames.append(frame)

    cap.release()

    frames = np.stack(frames)  # Convert list of frames to a NumPy array
    mean = frames.mean()
    std = frames.std()
    frames = (frames - mean) / std  # Normalize the frames

    # Convert to PyTorch tensor
    #frames = torch.from_numpy(frames).to(torch.float32)
    return frames

def load_alignments(path: str, char_to_num) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])

    # Convert the list of tokens to a single string
    text = ''.join(tokens)

    # Use the provided 'char_to_num' mapping to convert characters to numbers
    char_indices = [char_to_num[char] for char in text]

    # Remove the first element, which is ' ' (space)
    char_indices = char_indices[1:]

    return char_indices


def load_data(path: str):
    # Remove any potential leading or trailing white spaces
    path = path.strip()
    
    # Extract the file name
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    
    frames = load_video(video_path)  # Assuming you have a 'load_video' function
    alignments = load_alignments(alignment_path,char_to_num)  # Assuming you have a 'load_alignments' function
    
    
    return frames, alignments
