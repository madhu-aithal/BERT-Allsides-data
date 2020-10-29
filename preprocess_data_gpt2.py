import json_lines
import json
import pickle
import spacy
import csv
import os
import random
from pathlib import Path
import argparse

def get_allsides_desc_ideology_headline(data, item):
    # allsides_description <SEP> ideology (<LEFT>/<CENTER>/<RIGHT>) <SEP> news_headline
    
    if "allsides_description" in item and item["allsides_description"]:
        desc = item["allsides_description"].replace("\n", " ").strip()

        for article in item["articles"]:
            text_data = "<BOS> "+desc+" <SEP> "
            if article["political_spectrum"]:
                text_data += "<"+article["political_spectrum"].upper().strip()+"> "+article["article_headline"].replace("\n"," ").strip()+" <EOS>"
            data.append(text_data)
    return data

def get_allsides_desc_news_desc_ideology_headline(data, item):
    # allsides_description <SEP> news_description <SEP> ideology (<LEFT>/<CENTER>/<RIGHT>) news_headline

    if "allsides_description" in item and item["allsides_description"]:
        desc = item["allsides_description"].replace("\n", " ").strip()

        for article in item["articles"]:
            text_data = "<BOS> "+desc+" <SEP> "
            if article["political_spectrum"] and article["article_description"]:
                text_data += article["article_description"].replace("\n", " ") + "<SEP> <"+article["political_spectrum"].upper().strip()+"> "+article["article_headline"].replace("\n"," ").strip()+" <EOS>"
            data.append(text_data)

    return data

def write_to_text_file(out_file, data):
    with open(out_file, "w") as f:
        for d in data:
            f.write(d+"\n")      

def write_to_text_file_util(data, out_dir, file_prefix):
    train_size = int(0.9*len(data))
    write_to_text_file(os.path.join(out_dir, file_prefix+".txt"), data)
    write_to_text_file(os.path.join(out_dir, file_prefix+"_train.txt"), data[:train_size])
    write_to_text_file( os.path.join(out_dir, file_prefix+"_val.txt"), data[train_size:])

def read_samples_description_sep_headline(in_file, out_dir, args): 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    headline_desc_pair_data = []

    # desc_sep_headline_output_file = open(os.path.join(out_dir, "description_sep_headline.txt"), "w")
    desc_sep_headline_output_file_prefix = "description_sep_headline"
    allsides_desc_sep_LCR_headline_output_file_prefix = "allsides_desc_sep_LCR_headline"
    allsides_desc_news_desc_ideology_headline_output_file_prefix = "allsides_desc_news_desc_ideology_headline"

    desc_sep_headline_data = []
    allsides_desc_sep_ideology_headline_data = []
    allsides_desc_news_desc_ideology_headline_data = []

    with json_lines.open(in_file) as f:
        for item in f:
            for article in item["articles"]:
                desc_sep_headline_data.append("<BOS> "+article["article_description"].lower().replace("\n", " ") +" <SEP> "+article["article_headline"].replace("\n", " ")+" <EOS>")

            allsides_desc_sep_ideology_headline_data = get_allsides_desc_ideology_headline(allsides_desc_sep_ideology_headline_data, item)            
            allsides_desc_news_desc_ideology_headline_data = get_allsides_desc_news_desc_ideology_headline(allsides_desc_news_desc_ideology_headline_data, item)

        write_to_text_file_util(desc_sep_headline_data, \
            out_dir, \
            desc_sep_headline_output_file_prefix)

        write_to_text_file_util(allsides_desc_sep_ideology_headline_data, \
            out_dir, \
            allsides_desc_sep_LCR_headline_output_file_prefix)

        write_to_text_file_util(allsides_desc_news_desc_ideology_headline_data, \
            out_dir, \
            allsides_desc_news_desc_ideology_headline_output_file_prefix)        

if __name__=='__main__':
    TRAIN_DATASET_FILE_PATH = '/data/madhu/allsides_scraped_data/new_data_oct_7/full_data_train.jl.gz'
    SAVES_DIR = '/data/madhu/allsides_scraped_data/new_data_oct_7/processed_data_gpt2'

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_type",
                        default="article_headline", # Change this to None in future and make required = True
                        type=str,
                        required=False,
                        help="")    

    args = parser.parse_args()
    read_samples_description_sep_headline(TRAIN_DATASET_FILE_PATH, SAVES_DIR, args)
