import argparse
import pickle
import os
import random
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--dataset_file",
                        default="", # change default to None
                        type=str,
                        required=True, # change required to True
                        help="File location of processed dataset file (.pickle)")    
    parser.add_argument("--output_file",
                        default="", # change default to None
                        type=str,
                        required=True, # change required to True
                        help="Output file location in which you want to write sampled data") 
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        required=True,
                        help="No of samples to sample randomly")    
    # parser.add_argument("--article_type",
    #                     default="",
    #                     type=str,
    #                     required=True,
    #                     help="Whether domestic or international article you want to sample")    
    parser.add_argument("--LCR",                    
                    dest='lcr',
                    action='store_true',
                    help="Whether the dataset contains Left-Right data or Left-Center-Right data")

    args = parser.parse_args()
    parser.set_defaults(lcr=False)

    data = pickle.load(open(args.dataset_file, "rb"))

    if args.lcr:
        article_type_index = 4
    else:
        article_type_index = 3

    domestic_data = [val for val in data if val[article_type_index]=="domestic"]
    international_data = [val for val in data if val[article_type_index]=="international"]

    random.shuffle(domestic_data)
    sampled_data = domestic_data[:args.n_samples]

    random.shuffle(international_data)
    sampled_data += international_data[:args.n_samples]
    
    
    with open(args.output_file, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        if args.lcr:
            fields = ["", "", "", "correct_answer_index", "article_type"]
        else:
            fields = ["", "", "correct_answer_index", "article_type"]

        csvwriter.writerow(fields)
        for sample in sampled_data:
            csvwriter.writerow(sample)  
'''
python sample_articles.py --article_type "international" \
    --n_samples 10 \
    --dataset_file "/data/madhu/allsides_scraped_data/new_data_oct_7/processed_data_with_article_type/left_labeled_article_headline_LR.pickle" \
    --output_file "left_domestic_samples.csv"


python sample_articles.py --n_samples 10 --dataset_file "/data/madhu/allsides_scraped_data/new_data_oct_7/processed_data_with_article_type/right_labeled_article_headline_LR.pickle" --output_file "right_samples.csv"
'''

