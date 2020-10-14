import json_lines
import json
import pickle
import spacy
import csv
import os
import random
from pathlib import Path
import argparse

# nlp = spacy.load("en_core_web_md")

# def remove_newline(data):
#     new_data = []
#     for article in data:
#         for val in article:
#             10


def read_samples_util_LR(key, articles, left_labeled_data_LR, right_labeled_data_LR):
    left = [val for val in articles if val['political_spectrum'] == 'Left']
    right = [val for val in articles if val['political_spectrum'] == 'Right']

    for val1 in left:
        for val2 in right:
            data = [val1[key], val2[key]]
            random.shuffle(data)
            
            data_left = data + [data.index(val1[key])]
            data_right = data + [data.index(val2[key])]

            left_labeled_data_LR.append(data_left)
            right_labeled_data_LR.append(data_right)

    return left_labeled_data_LR, right_labeled_data_LR

def read_samples_util_LCR(key, articles, left_labeled_data_LCR, right_labeled_data_LCR):
    left = [val for val in articles if val['political_spectrum'] == 'Left']
    center = [val for val in articles if val['political_spectrum'] == 'Center']
    right = [val for val in articles if val['political_spectrum'] == 'Right']

    for val1 in left:
        for val2 in right:
            for val3 in center:
                data = [val1[key], val2[key], val3[key]]
                random.shuffle(data)
                
                data_left = data + [data.index(val1[key])]                
                data_right = data + [data.index(val2[key])]

                left_labeled_data_LCR.append(data_left)
                right_labeled_data_LCR.append(data_right)

    return left_labeled_data_LCR, right_labeled_data_LCR



# Function for creating data in Left headline, Center description, Right headline
def read_samples_util_Left_Center_Desc_Right(key, articles, left_labeled_data_LCR, right_labeled_data_LCR, article_type):
    left = [val for val in articles if val['political_spectrum'] == 'Left']
    center = [val for val in articles if val['political_spectrum'] == 'Center']
    right = [val for val in articles if val['political_spectrum'] == 'Right']

    for val1 in left:
        for val2 in right:
            for val3 in center:
                data = [val1[key], val2[key], val3['article_description']]
                random.shuffle(data)
                # data = data + []
                data_left = data + [data.index(val1[key]), article_type]                
                data_right = data + [data.index(val2[key]), article_type]

                left_labeled_data_LCR.append(data_left)
                right_labeled_data_LCR.append(data_right)

    return left_labeled_data_LCR, right_labeled_data_LCR

def read_samples(in_file, out_dir, args):    
    left_labeled_data_LR = []
    right_labeled_data_LR = []

    # LCR version stores [Left Headline, Center description, Right headline]
    left_labeled_data_LCR = []
    right_labeled_data_LCR = []
    with json_lines.open(in_file) as f:
        for item in f:                                    
            left_labeled_data_LR, right_labeled_data_LR = read_samples_util_LR(args.data_type, item['articles'], left_labeled_data_LR, right_labeled_data_LR)            
            
            # left_labeled_data_LCR, right_labeled_data_LCR = read_samples_util_LCR(args.data_type, item['articles'], left_labeled_data_LCR, right_labeled_data_LCR)            
            
            # left_labeled_data_LCR, right_labeled_data_LCR = read_samples_util_Left_Center_Desc_Right("article_headline", item['articles'], left_labeled_data_LCR, right_labeled_data_LCR)            

        Path(out_dir).mkdir(parents=True, exist_ok=True)           
        
        pickle.dump(left_labeled_data_LR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LR.pickle"), 'wb'))
        pickle.dump(right_labeled_data_LR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LR.pickle"), 'wb'))

        # pickle.dump(left_labeled_data_LCR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LCR.pickle"), 'wb'))
        # pickle.dump(right_labeled_data_LCR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LCR.pickle"), 'wb'))

        # pickle.dump(left_labeled_data_LCR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LCR_label_count_3.pickle"), 'wb'))
        # pickle.dump(right_labeled_data_LCR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LCR_label_count_3.pickle"), 'wb'))


if __name__=='__main__':
    TRAIN_DATASET_FILE_PATH = '/data/madhu/allsides_scraped_data/new_data_oct_7/full_data_train.jl.gz'
    SAVES_DIR = '/data/madhu/allsides_scraped_data/new_data_oct_7/processed_data'

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_type",
                        default="article_headline", # Change this to None in future and make required = True
                        type=str,
                        required=False,
                        help="")    

    args = parser.parse_args()
    read_samples(TRAIN_DATASET_FILE_PATH, SAVES_DIR, args)
