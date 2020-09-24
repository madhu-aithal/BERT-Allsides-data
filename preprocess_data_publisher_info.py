import json_lines
import json
import pickle
import spacy
import csv
import os
import numpy as np

nlp = spacy.load("en_core_web_md")

def read_samples(file):
    data_dir = os.path.dirname(file)

    # headlines = {
    #     'left': [],
    #     'center': [],
    #     'right': []
    # }
    # descriptions = {
    #     'left': [],
    #     'center': [],
    #     'right': []
    # }
    headlines = []
    descriptions = []
    publisher_spectrum_map = {}
    label_map = {
        'left': 0,
        'center': 1,
        'right': 2
    }
    counts = {
        'left': 0,
        'center': 0,
        'right': 0
    }
    split_ratio = 0.9
    with json_lines.open(file) as f:
        for item in f:        
                            

            headline = " ".join([val.lower() for val in item["article_headline"].split()])
            desc = " ".join([val.lower() for val in item["article_description"].split()])
            # print(item['publisher'])
            news_publisher = None
            if item['publisher'] != None:
                news_publisher = " ".join([val.lower() for val in item["publisher"].split()])

            if item['publisher'] != None:
                if item['publisher'] not in publisher_spectrum_map:
                    publisher_spectrum_map[item['publisher']] = []   
                
                publisher_spectrum_map[item['publisher']].append(item["political_spectrum"].lower())            


            if item["political_spectrum"].lower() != "" and item['publisher'] != None:
                headlines.append([news_publisher+". "+headline, label_map[item["political_spectrum"].lower()]])
                descriptions.append([news_publisher+". "+desc, label_map[item["political_spectrum"].lower()]])
                # headlines[item["political_spectrum"].lower()].append(news_publisher+". "+headline)
                # descriptions[item["political_spectrum"].lower()].append(news_publisher+". "+desc)
                # counts[item["political_spectrum"].lower()] += 1
            # except:
                # print("Val: ", item["political_spectrum"].lower())
        
        # train_headlines = []
        # test_headlines = []
        
        # train_descriptions = []
        # test_descriptions = []
        # for label in ["left", "center", "right"]:
        #     train_size = int(len(headlines[label])*split_ratio)
        #     for val in headlines[label][:train_size]:
        #         train_headlines.append([val, label_map[label]])

        #     for val in headlines[label][train_size:]:
        #         test_headlines.append([val, label_map[label]])


        #     train_size = int(len(descriptions[label])*split_ratio)
        #     for val in descriptions[label][:train_size]:
        #         train_descriptions.append([val, label_map[label]])
                
        #     for val in descriptions[label][train_size:]:
        #         test_descriptions.append([val, label_map[label]])

        pickle.dump(headlines, open(os.path.join(data_dir, 'article_headlines_train_with_publisher.pickle'), 'wb'))
        # pickle.dump(test_headlines, open(os.path.join(data_dir, 'article_headlines_test_with_publisher.pickle'), 'wb'))
        pickle.dump(descriptions, open(os.path.join(data_dir, 'article_descriptions_train_with_publisher.pickle'), 'wb'))
        # pickle.dump(test_descriptions, open(os.path.join(data_dir, 'article_descriptions_test_with_publisher.pickle'), 'wb'))

        for key,val in publisher_spectrum_map.items():
            # print(key, np.unique(np.array(val)))
            if len(np.unique(np.array(val))) > 1:
                print(key, np.unique(np.array(val)))

if __name__=='__main__':
    DATASET_FILE_PATH = '/data/madhu/allsides_scraped_data/full_data_train.jl.gz'

    read_samples(DATASET_FILE_PATH)