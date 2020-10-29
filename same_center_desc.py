import pickle
import random
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--dataset_file",
                        default="", # change default to None
                        type=str,
                        required=True, # change required to True
                        help="File location of processed dataset file (.pickle)")    
    parser.add_argument("--output_filename",
                        default="", # change default to None
                        type=str,
                        required=True, # change required to True
                        help="Output filename (just filename is enough, no need of absolute path) in which you want to write same center description added data") 
    
    # Randomly picked center description from left labeled pickle file
    center_description = 'President Donald Trump said on Tuesday he has declassified all documents related to federal investigations into Russian election interference and former Secretary of State Hillary Clinton’s use of a private server for government emails.\n “I have fully authorized the total Declassification of any & all documents pertaining to the single greatest political CRIME in American History, the Russia Hoax. Likewise, the Hillary Clinton Email Scandal. No redactions!” Trump wrote on Twitter.'

    args = parser.parse_args()

    out_dir = os.path.dirname(args.dataset_file)

    data = pickle.load(open(args.dataset_file, "rb"))

    data_with_same_center_desc = []
    for val in data:
        LR_data = val[:2]
        label_data = val[val[2]]
        temp = LR_data+[center_description]
        random.shuffle(temp)
        temp = temp + [temp.index(label_data)]+[val[-1]]
        
        data_with_same_center_desc.append(temp)

    pickle.dump(data_with_same_center_desc, open(os.path.join(out_dir, args.output_filename), "wb"))

    