import pickle
from os import listdir
from os.path import isfile, join
import json_lines
import random
import argparse
from pathlib import Path
import os
from preprocess_data import read_samples_util_Left_Center_Desc_Right
# import spacy

# nlp = spacy.load("en_core_web_md")

def load_country_names_and_demonyms(keywords_dir):
    keywords = set()

    onlyfiles = [f for f in listdir(keywords_dir) if isfile(join(keywords_dir, f))]
    print(onlyfiles)
    for filepath in onlyfiles:
        with open(os.path.join(keywords_dir,filepath), "r") as fin:
            for line in fin:
                keywords.add(line.strip().lower())

    return keywords

def check_keywords(text, keywords):
    text = text.lower().replace("\n", " ")
    # doc = nlp(text)
    # for token in doc:
    for word in text.split():
        if word.strip() in keywords:
            return True    
    return False

def read_samples(in_file, international_keywords, domestic_keywords):    
    
    domestic_articles = []
    international_articles = []

    with json_lines.open(in_file) as f:
        count = 0
        for item in f:   
            international_flag = False
            count += 1
            if item['allsides_description']:
                international_flag = international_flag or check_keywords(item['allsides_description'], international_keywords)                
            
            for article in item['articles']:
                if article['article_description']:
                    international_flag = international_flag or check_keywords(article['article_description'], international_keywords)
            
            if check_keywords(item['allsides_description'], domestic_keywords):
                international_flag = False

            for article in item['articles']:
                if article['article_description']:
                    if check_keywords(article['article_description'], domestic_keywords):
                        international_flag = False

            if not international_flag:
                item["type"] = "domestic"
                domestic_articles.append(item)
            else:
                item["type"] = "international"
                international_articles.append(item)
        # print(count)
        return domestic_articles, international_articles


def read_samples_util_LR(key, articles, left_labeled_data_LR, right_labeled_data_LR, article_type):
    left = [val for val in articles if val['political_spectrum'] == 'Left']
    right = [val for val in articles if val['political_spectrum'] == 'Right']

    for val1 in left:
        for val2 in right:
            data = [val1[key], val2[key]]
            random.shuffle(data)
            
            data_left = data + [data.index(val1[key])] + [article_type]
            data_right = data + [data.index(val2[key])] + [article_type]

            left_labeled_data_LR.append(data_left)
            right_labeled_data_LR.append(data_right)

    return left_labeled_data_LR, right_labeled_data_LR

def read_samples_util_LCR(key, articles, left_labeled_data_LCR, right_labeled_data_LCR, article_type):
    left = [val for val in articles if val['political_spectrum'] == 'Left']
    center = [val for val in articles if val['political_spectrum'] == 'Center']
    right = [val for val in articles if val['political_spectrum'] == 'Right']

    for val1 in left:
        for val2 in right:
            for val3 in center:
                data = [val1[key], val2[key], val3[key]]
                random.shuffle(data)
                
                data_left = data + [data.index(val1[key])] + [article_type]             
                data_right = data + [data.index(val2[key])] + [article_type]

                left_labeled_data_LCR.append(data_left)
                right_labeled_data_LCR.append(data_right)

    return left_labeled_data_LCR, right_labeled_data_LCR

def process_selected_samples(articles, out_dir, args):    
    left_labeled_data_LR = []
    right_labeled_data_LR = []
    left_labeled_data_LCR = []
    right_labeled_data_LCR = []
    left_labeled_data_L_CenterDesc_R = []
    right_labeled_data_L_CenterDesc_R = []
    
    for item in articles:                                    
        left_labeled_data_LR, right_labeled_data_LR = read_samples_util_LR(args.data_type, item['articles'], left_labeled_data_LR, right_labeled_data_LR, item["type"])            
        # left_labeled_data_LCR, right_labeled_data_LCR = read_samples_util_LCR(args.data_type, item['articles'], left_labeled_data_LCR, right_labeled_data_LCR)                    
        left_labeled_data_L_CenterDesc_R, right_labeled_data_L_CenterDesc_R = read_samples_util_Left_Center_Desc_Right(args.data_type, item['articles'], left_labeled_data_L_CenterDesc_R, right_labeled_data_L_CenterDesc_R, item["type"])

    Path(out_dir).mkdir(parents=True, exist_ok=True)           

    print("left_labeled_data_LR: ", len(left_labeled_data_LR))
    print("right_labeled_data_LR: ", len(right_labeled_data_LR))
    print("left_labeled_data_LCR: ", len(left_labeled_data_LCR))
    print("right_labeled_data_LCR: ", len(right_labeled_data_LCR))

    pickle.dump(left_labeled_data_LR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LR.pickle"), 'wb'))
    pickle.dump(right_labeled_data_LR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LR.pickle"), 'wb'))

    # pickle.dump(left_labeled_data_LCR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LCR.pickle"), 'wb'))
    # pickle.dump(right_labeled_data_LCR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LCR.pickle"), 'wb'))

    pickle.dump(left_labeled_data_LCR, open(os.path.join(out_dir, "left_labeled_"+args.data_type+"_LCR_label_count_3.pickle"), 'wb'))
    pickle.dump(right_labeled_data_LCR, open(os.path.join(out_dir, "right_labeled_"+args.data_type+"_LCR_label_count_3.pickle"), 'wb'))



# Rename functions to make their purpose more clear
if __name__=="__main__":
    TRAIN_DATASET_FILE_PATH = '/data/madhu/allsides_scraped_data/new_data_oct_7/full_data_train.jl.gz'

    parser = argparse.ArgumentParser()

    ## Required parameters
    
    parser.add_argument("--data_type",
                        default="article_headline", # change default to None
                        type=str,
                        required=False, # change required to True
                        help="")    

    args = parser.parse_args()

    international_keywords = load_country_names_and_demonyms("./international_keywords")
    domestic_keywords = load_country_names_and_demonyms("./domestic_keywords")
    domestic_articles, international_articles = read_samples(TRAIN_DATASET_FILE_PATH, international_keywords, domestic_keywords)
    print("domestic_articles: ", len(domestic_articles))
    print("international_articles: ", len(international_articles))
    # print(domestic_articles)
    out_dir = os.path.join(os.path.dirname(TRAIN_DATASET_FILE_PATH), "processed_data_with_article_type")
    process_selected_samples(domestic_articles+international_articles, out_dir, args)

    # out_dir = os.path.join(os.path.dirname(TRAIN_DATASET_FILE_PATH), "processed_data_international_articles")
    # process_selected_samples(international_articles, out_dir, args)
