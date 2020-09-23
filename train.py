import torch
import os
import sys
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import time
import random
from pathlib import Path
import utils
import datetime
import logging
from torch import nn
import argparse
import pprint
import json_lines
import random

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
    
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))

# def read_file(dataset_path: str):
    
            # print(item['x'])

# def collate_fn(data):
#     print(data)

def train_model(args: dict, hparams:dict):
    # Code for this function adopted from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    
    file = args.dataset_filepath
    # pos_file = args.pos_file
    # neg_file = args.neg_file
    truncation = args.truncation
    # n_samples = args.n_samples
    seed_val = hparams["seed_val"]
    device = utils.get_device(device_no=args.device_no)
    saves_dir = "saves/"

    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    time = datetime.datetime.now()

    saves_path = os.path.join(saves_dir, utils.get_filename(time))
    Path(saves_path).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(saves_path, "training.log")

    logging.basicConfig(filename=log_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger=logging.getLogger()
    # logger.setLevel()

    logger.info("File: "+str(file))
    logger.info("Parameters: "+str(args))
    logger.info("Truncation: "+truncation)

    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0

    
    # samples = utils.read_and_sample(file
    # # , seed_val=seed_val
    # )
    samples = utils.read_pairwise(file, first=0, second=2)
    

    random.shuffle(samples)
    input_ids = []
    attention_masks = []
    
    samples_text = [val[0] for val in samples]
    samples_label = [val[1] for val in samples]

    print(np.unique(np.array(samples_label)))

    max_len = 0

    # For every sentence...
    for text in samples_text:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_id = tokenizer(text, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_id['input_ids']))

    logger.info('Max text length: ' + str(max_len))
    print('Max text length: ' + str(max_len))

    for text in samples_text:        
        input_id = tokenizer(text, add_special_tokens=True)
        # print(len(input_id['input_ids']))
        # if len(input_id['input_ids']) > 512:                        
        #     if truncation == "tail-only":
        #         input_id = [tokenizer.cls_token_id]+input_id[-511:]      
        #     elif truncation == "head-and-tail":
        #         input_id = [tokenizer.cls_token_id]+input_id[1:129]+input_id[-382:]+[tokenizer.sep_token_id]
        #     else:
        #         input_id = input_id[:511]+[tokenizer.sep_token_id]
                
        #     input_ids.append(torch.tensor(input_id).view(1,-1))
        #     attention_masks.append(torch.ones([1,len(input_id)], dtype=torch.long))
        # else:
        encoded_dict = tokenizer(
                            text,                      
                            add_special_tokens = True, 
                            truncation=True,                               
                            max_length = 512,         
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                    )
                    
        input_ids.append(encoded_dict['input_ids'])
                    
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    samples_label_tensor = torch.tensor(samples_label)
    # samples_text_tensor = torch.tensor(samples_text)
    
    dataset = TensorDataset(input_ids, attention_masks, samples_label_tensor)
    # dataset = TensorDataset(samples_text_tensor, samples_label_tensor)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info('{:>5,} training samples'.format(train_size))
    logger.info('{:>5,} validation samples'.format(val_size))

    batch_size = hparams["batch_size"]

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                # collate_fn = collate_fn
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Evaluate with this batch size.
                # collate_fn = collate_fn
            )


    model = BertForSequenceClassification.from_pretrained(        
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.        
    )
    
    
    model = model.to(device=device)    
    # model.cuda(device=device)

    optimizer = AdamW(model.parameters(),
                    lr = hparams["learning_rate"], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = hparams["adam_epsilon"] # args.adam_epsilon  - default is 1e-8.
                    )
    epochs = 4

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    for epoch_i in range(0, epochs):
        
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')

        
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            print(len(train_dataloader))
        
            if step % 40 == 0 and not step == 0:               
                logger.info('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)           

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            # print(logits)
            # print(loss)
            
            total_train_loss += loss.detach().cpu().numpy()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)            

        logger.info("")
        logger.info("Average training loss: {0:.2f}".format(avg_train_loss))

            
        logger.info("")
        logger.info("Running Validation...")

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        

                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            total_eval_loss += loss.detach().cpu().numpy()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logger.info("Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
               
        logger.info("Validation Loss: {0:.2f}".format(avg_val_loss))        

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,                
            }
        )

        model_save_path = os.path.join(saves_path, "model_"+str(epoch_i+1)+"epochs")
        torch.save(model, model_save_path)

    logger.info("")
    logger.info("Training complete!")
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__=="__main__":   
    

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    parser.add_argument("--device_no",
                        default=0,
                        type=int,
                        help="")
    parser.add_argument("--truncation",
                    default=None,
                    type=str,
                    required=True,
                    help="Possible values: head-only, tail-only, and head-and-tail")

    args = parser.parse_args()

    hyperparams = [       
        {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "seed_val": 23,
            "adam_epsilon": 1e-8
        },  
    ]

    for hparams in hyperparams:         
        myprint(f"args: {args}")    
        myprint(f"hparams: {hparams}")
        train_model(args, hparams)
        

    