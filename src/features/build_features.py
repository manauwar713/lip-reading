import os
import cv2
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import dlib
import torch
import pathlib
import yaml
import json
import pickle

shape_predictor_path = "/raid/syscon/malam/lip/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)


def load_video(path,margin):
    path_train = os.path.join(path,'train','s1')
    path_test = os.path.join(path,'test','s1')
    frames_train_dict = {}
    frames_test_dict = {}
    for file in os.listdir(path_train):
        
        cap = cv2.VideoCapture(os.path.join(path_train,file))
        

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 75:
            continue
        
        frames_train = []
        
        for _ in range(frame_count):
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)
            if len(faces) == 0:
                
                break
            face = faces[0]

            # Define custom cropping parameters for the mouth region and the area around it
            x, y, w, h = shape_predictor(gray_frame, face).part(48).x, shape_predictor(gray_frame, face).part(49).y, shape_predictor(gray_frame, face).part(54).x - shape_predictor(gray_frame, face).part(48).x, shape_predictor(gray_frame, face).part(57).y - shape_predictor(gray_frame, face).part(49).y

            # Extend the cropping parameters to include areas around the mouth
             # Adjust this value to control the size of the extracted area
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin

            # Make sure the cropping area stays within the frame boundaries
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, gray_frame.shape[1] - x)
            h = min(h, gray_frame.shape[0] - y)

            # Extract the enlarged mouth region
            mouth_region = gray_frame[y:y+h, x:x+w]
            mouth_region = cv2.resize(mouth_region, (100,50))
            
            
            frames_train.append(mouth_region)
            

            # Display the enlarged mouth region using Matplotlib
            # plt.imshow(mouth_region)
            # plt.title(f'Enlarged Mouth Region')
            # plt.show()
            # print(mouth_region.shape)

            
        cap.release()
        basename = os.path.basename(os.path.join(path_train,file))
        code_train = os.path.splitext(basename)[0]

        # if not frames:
        #     raise ValueError("No frames loaded from the video.")
        if len(frames_train) == 0:
            continue
        else:
            stack = np.stack(frames_train)
            mean = stack.mean()
            std = stack.std()
            stack = (stack - mean) / std
            frames_train_dict[code_train] = stack
        
    for file in os.listdir(path_test):
        cap = cv2.VideoCapture(os.path.join(path_test,file))
        

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 75:
            continue
        
        frames_test = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)
            if len(faces) == 0:
                
                break
            face = faces[0]

            # Define custom cropping parameters for the mouth region and the area around it
            x, y, w, h = shape_predictor(gray_frame, face).part(48).x, shape_predictor(gray_frame, face).part(49).y, shape_predictor(gray_frame, face).part(54).x - shape_predictor(gray_frame, face).part(48).x, shape_predictor(gray_frame, face).part(57).y - shape_predictor(gray_frame, face).part(49).y

            # Extend the cropping parameters to include areas around the mouth
             # Adjust this value to control the size of the extracted area
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin

            # Make sure the cropping area stays within the frame boundaries
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, gray_frame.shape[1] - x)
            h = min(h, gray_frame.shape[0] - y)

            # Extract the enlarged mouth region
            mouth_region = gray_frame[y:y+h, x:x+w]
            mouth_region = cv2.resize(mouth_region, (100,50))
            
            
            frames_test.append(mouth_region)

            # Display the enlarged mouth region using Matplotlib
            # plt.imshow(mouth_region)
            # plt.title(f'Enlarged Mouth Region')
            # plt.show()
            # print(mouth_region.shape)

            
        cap.release()
        basename = os.path.basename(os.path.join(path_test,file))
        code_test = os.path.splitext(basename)[0]

        # if not frames:
        #     raise ValueError("No frames loaded from the video.")

        if len(frames_test) == 0:
            continue
        else:
            stack = np.stack(frames_test)
            mean = stack.mean()
            std = stack.std()
            stack = (stack - mean) / std
            frames_test_dict[code_test] = stack

    return frames_train_dict,frames_test_dict

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = {char: num for num, char in enumerate(vocab)}
char_to_num[""] = len(vocab) 
num_to_char = {num: char for char, num in char_to_num.items()}

def load_alignments(path: str,frames_train,frames_test):
    path_train = os.path.join(path,'train','align')
    path_test = os.path.join(path,'test','align')
    char_test = {}
    char_train = {}
    for file in os.listdir(path_train):
        
        basename_train = os.path.basename(os.path.join(path_train,file))
        code_train = os.path.splitext(basename_train)[0]
        
        if code_train in frames_train:
            print(code_train)
            with open(os.path.join(path_train,file), 'r') as f:
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
            
            char_train[code_train] = char_indices
    for file in os.listdir(path_test):
        print(file)
        basename_test = os.path.basename(os.path.join(path_test,file))
        code_test = os.path.splitext(basename_test)[0]
        if code_test in frames_test:
            print(code_test)
            with open(os.path.join(path_test,file), 'r') as f:
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
            
            char_test[code_test] = char_indices

    return char_train,char_test


def save_data(frames_train,frames_test,alignments_train,alignments_test,output_path):
    
    
    train_data = {}
    for code,frames in frames_train.items():
        train_data[code] = {}
        train_data[code]['frame'] = frames
    for code,align in alignments_train.items():
        train_data[code]['alignments'] = align
    test_data = {}
    for code,frames in frames_test.items():
        test_data[code] = {}
        test_data[code]['frame'] = frames
    for code,align in alignments_test.items():
        test_data[code]['alignments'] = align
        
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train_dir = os.path.join(output_path,'train.pkl')
    test_dir = os.path.join(output_path,'test.pkl')
     
    
    
    with open(train_dir,'wb') as f:
        pickle.dump(train_data,f)
    with open(test_dir,'wb') as f:
        pickle.dump(test_data,f)  
    

    

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["build_feature"]
    
    
    data_path = home_dir.as_posix() + '/data/processed'
    output_path = home_dir.as_posix() + '/data/interim'
    
    frames_train,frames_test = load_video(data_path,params['margin'])
    
    char_train,char_test = load_alignments(data_path,frames_train,frames_test)
    print(char_train)
    print(char_test)
    save_data(frames_train,frames_test,char_train,char_test,output_path)
    print(char_to_num)
    
if __name__ == '__main__':
    
    main()
    
