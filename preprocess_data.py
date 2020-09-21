import json_lines
import json
import pickle
import spacy
import csv


nlp = spacy.load("en_core_web_md")

def read_samples(file):
    headlines = {
        'left': [],
        'center': [],
        'right': []
    }
    descriptions = {
        'left': [],
        'center': [],
        'right': []
    }
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
            # try:
            if item["political_spectrum"].lower() != "":
                # headlines.append((headline, label_map[item["political_spectrum"].lower()]))
                # descriptions.append((desc, label_map[item["political_spectrum"].lower()]))
                headlines[item["political_spectrum"].lower()].append(headline)
                descriptions[item["political_spectrum"].lower()].append(desc)
                # counts[item["political_spectrum"].lower()] += 1
            # except:
                # print("Val: ", item["political_spectrum"].lower())
        
        train_headlines = []
        test_headlines = []
        
        train_descriptions = []
        test_descriptions = []
        for label in ["left", "center", "right"]:
            train_size = int(len(headlines[label])*split_ratio)
            for val in headlines[label][:train_size]:
                train_headlines.append([val, label_map[label]])

            for val in headlines[label][train_size:]:
                test_headlines.append([val, label_map[label]])


            train_size = int(len(descriptions[label])*split_ratio)
            for val in descriptions[label][:train_size]:
                train_descriptions.append([val, label_map[label]])
                
            for val in descriptions[label][train_size:]:
                test_descriptions.append([val, label_map[label]])

        pickle.dump(train_headlines, open('article_headlines_train.pickle', 'wb'))
        pickle.dump(test_headlines, open('article_headlines_test.pickle', 'wb'))
        pickle.dump(train_descriptions, open('article_descriptions_train.pickle', 'wb'))
        pickle.dump(test_descriptions, open('article_descriptions_test.pickle', 'wb'))

        # with open('headlines_train.csv','w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['headline', 'label'])
        #     writer.writerows(train_headlines)
        
        # with open('headlines_test.csv','w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['headline', 'label'])
        #     writer.writerows(test_headlines)

        # with open('descriptions_train.csv','w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['description', 'label'])
        #     writer.writerows(train_descriptions)
        
        # with open('descriptions_test.csv','w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['description', 'label'])
        #     writer.writerows(test_descriptions)
                
        # print(len(headlines))
        # print(len(descriptions))
        # print(counts)
        # train_size = 0.
        # train_headlines = 
        # pickle.dump(headlines, open('article_headlines.pickle', 'wb'))
        # pickle.dump(descriptions, open('article_descriptions.pickle', 'wb'))

if __name__=='__main__':
    DATASET_FILE_PATH = '/data/madhu/allsides_scraped_data/full_data.jl.gz'

    read_samples(DATASET_FILE_PATH)