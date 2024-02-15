import pathlib
import yaml
import sys
import os
import shutil
from sklearn.model_selection import train_test_split

def load_data(path,train_split,data_part,seed):
    alignment_dir = os.path.join(path,'alignments/s1')
    s1_dir = os.path.join(path,'s1')
    alignment_files = sorted([file for file in os.listdir(alignment_dir) if file.endswith('.align')])
    #alignment_code = [file.split(".")[0] for file in alignment_files]
    s1_files = sorted([file for file in os.listdir(s1_dir) if file.endswith('.mpg')])
    
        
        
    
    data_align = alignment_files[:data_part]
     
    data_s1 = s1_files[:data_part]
    
    slicing = int(train_split*data_part)
    train_align,test_align = data_align[:slicing],data_align[slicing:]
    train_s1,test_s1 = data_s1[: slicing],data_s1[slicing:]
    return train_align,test_align,train_s1,test_s1
    
    
    
def save_data(train_align,test_align,train_s1,test_s1,output_path,path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train_dir_align = os.path.join(output_path,'train','align')
    train_dir_s1 = os.path.join(output_path,'train','s1')
    test_dir_align = os.path.join(output_path,'test','align')
    test_dir_s1 = os.path.join(output_path,'test','s1') 
    pathlib.Path(train_dir_align).mkdir(parents=True,exist_ok=True)
    pathlib.Path(train_dir_s1).mkdir(parents=True,exist_ok=True)
    pathlib.Path(test_dir_align).mkdir(parents=True,exist_ok=True)
    pathlib.Path(test_dir_s1).mkdir(parents=True,exist_ok=True)
    
    for file_name in train_align:
        src = os.path.join(path,'alignments/s1',file_name)
        dst = os.path.join(train_dir_align,file_name)
        shutil.copy(src,dst)
        
    for file_name in test_align:
        src = os.path.join(path,'alignments/s1',file_name)
        dst = os.path.join(test_dir_align,file_name)
        shutil.copy(src,dst)
    
    for file_name in train_s1:
        src = os.path.join(path,'s1',file_name)
        dst = os.path.join(train_dir_s1,file_name)
        shutil.copy(src,dst)
        
    for file_name in test_s1:
        src = os.path.join(path,'s1',file_name)
        dst = os.path.join(test_dir_s1,file_name)
        shutil.copy(src,dst)
    
    

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    
    data_path = home_dir.as_posix() + '/data/raw/data'
    output_path = home_dir.as_posix() + '/data/processed'
    
    train_align,test_align,train_s1,test_s1 = load_data(data_path,params['train_split'],params['data_part'],params['seed'])
    save_data(train_align,test_align,train_s1,test_s1,output_path,data_path)



if __name__ == '__main__':
    
    main()
