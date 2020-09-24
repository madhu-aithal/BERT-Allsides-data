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


def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval 
    based on the magnitude of the total iterations.

    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the 
                              course of the for-loop.
    '''
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller. 
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval

def make_smart_batches(text_samples, labels, batch_size, logger, tokenizer, max_len):
    '''
    This function combines all of the required steps to prepare batches.
    '''

    # print('Creating Smart Batches from {:,} examples with batch size {:,}...\n'.format(len(text_samples), batch_size))
    logger.info('Creating Smart Batches from {:,} examples with batch size {:,}...\n'.format(len(text_samples), batch_size))

    # =========================
    #   Tokenize & Truncate
    # =========================

    full_input_ids = []

    # Tokenize all training examples
    # print('Tokenizing {:,} samples...'.format(len(labels)))
    logger.info('Tokenizing {:,} samples...'.format(len(labels)))

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(labels), num_desired_updates=10)

    # For each training example...
    for text in text_samples:
        
        # Report progress.
        if ((len(full_input_ids) % update_interval) == 0):
            # print('  Tokenized {:,} samples.'.format(len(full_input_ids)))
            logger.info('  Tokenized {:,} samples.'.format(len(full_input_ids)))

        # Tokenize the sample.
        input_ids = tokenizer(text=text,              # Text to encode.
                                    add_special_tokens=True, # Do add specials.
                                    max_length=max_len,      # Do Truncate!
                                    truncation=True,         # Do Truncate!
                                    padding=False)           # DO NOT pad.
                                    
        # Add the tokenized result to our list.
        full_input_ids.append(input_ids)
        
    # print('DONE.')    
    # print('{:>10,} samples\n'.format(len(full_input_ids)))

    logger.info('DONE.')    
    logger.info('{:>10,} samples\n'.format(len(full_input_ids)))


    # =========================
    #      Select Batches
    # =========================    

    # Sort the two lists together by the length of the input sequence.
    samples = sorted(zip(full_input_ids, labels), key=lambda x: len(x[0]))

    # print('{:>10,} samples after sorting\n'.format(len(samples)))
    logger.info('{:>10,} samples after sorting\n'.format(len(samples)))

    import random

    # List of batches that we'll construct.
    batch_ordered_sentences = []
    batch_ordered_labels = []

    # print('Creating batches of size {:}...'.format(batch_size))
    logger.info('Creating batches of size {:}...'.format(batch_size))

    # Choose an interval on which to print progress updates.
    update_interval = good_update_interval(total_iters=len(samples), num_desired_updates=10)
    
    # Loop over all of the input samples...    
    while len(samples) > 0:
        
        # Report progress.
        if ((len(batch_ordered_sentences) % update_interval) == 0 \
            and not len(batch_ordered_sentences) == 0):
            # print('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))
            logger.info('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))

        # `to_take` is our actual batch size. It will be `batch_size` until 
        # we get to the last batch, which may be smaller. 
        to_take = min(batch_size, len(samples))

        # Pick a random index in the list of remaining samples to start
        # our batch at.
        select = random.randint(0, len(samples) - to_take)

        # Select a contiguous batch of samples starting at `select`.
        #print("Selecting batch from {:} to {:}".format(select, select+to_take))
        batch = samples[select:(select + to_take)]

        #print("Batch length:", len(batch))

        # Each sample is a tuple--split them apart to create a separate list of 
        # sequences and a list of labels for this batch.
        batch_ordered_sentences.append([s[0] for s in batch])
        batch_ordered_labels.append([s[1] for s in batch])

        # Remove these samples from the list.
        del samples[select:select + to_take]

    # print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))
    logger.info('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))

    # =========================
    #        Add Padding
    # =========================    

    logger.info('Padding out sequences within each batch...')

    py_inputs = []
    py_attn_masks = []
    py_labels = []

    # For each batch...
    for (batch_inputs, batch_labels) in zip(batch_ordered_sentences, batch_ordered_labels):

        # New version of the batch, this time with padded sequences and now with
        # attention masks defined.
        batch_padded_inputs = []
        batch_attn_masks = []
        
        # First, find the longest sample in the batch. 
        # Note that the sequences do currently include the special tokens!
        max_size = max([len(sen['input_ids']) for sen in batch_inputs])

        # For each input in this batch...
        for sen in batch_inputs:
            
            # How many pad tokens do we need to add?
            num_pads = max_size - len(sen['input_ids'])

            # Add `num_pads` padding tokens to the end of the sequence.
            padded_input = sen['input_ids'] + [tokenizer.pad_token_id]*num_pads

            # Define the attention mask--it's just a `1` for every real token
            # and a `0` for every padding token.
            attn_mask = [1] * len(sen['attention_mask']) + [0] * num_pads

            # Add the padded results to the batch.
            batch_padded_inputs.append(padded_input)
            batch_attn_masks.append(attn_mask)

        # Our batch has been padded, so we need to save this updated batch.
        # We also need the inputs to be PyTorch tensors, so we'll do that here.
        # Todo - Michael's code specified "dtype=torch.long"
        py_inputs.append(torch.tensor(batch_padded_inputs))
        py_attn_masks.append(torch.tensor(batch_attn_masks))
        py_labels.append(torch.tensor(batch_labels))
    
    logger.info('  DONE.')

    # Return the smart-batched dataset!
    return (py_inputs, py_attn_masks, py_labels)

def train_model(args: dict, hparams:dict):
    
    file = args.dataset_filepath
    # truncation = args.truncation

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

    logger.info("File: "+str(file))
    logger.info("Parameters: "+str(args))
    logger.info("Hyperparameters: "+str(hparams))
    # logger.info("Truncation: "+truncation)

    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0

    if args.binary_classifier:        
        samples = utils.read_pairwise(file, args.data_1, args.data_2, dataset_amount=args.dataset_amount)
    else:
        samples = utils.read_and_sample(file, dataset_amount=args.dataset_amount)

    no_of_labels = len(np.unique(np.array([val[1] for val in samples])))

    logger.info("No of unique labels: "+str(no_of_labels))

    train_size = int(0.9 * len(samples))
    val_size = len(samples) - train_size

    random.shuffle(samples)

    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    train_samples_text = [val[0] for val in train_samples]
    train_samples_label = [val[1] for val in train_samples]
    val_samples_text = [val[0] for val in val_samples]
    val_samples_label = [val[1] for val in val_samples]

    max_len = 0

    # For every sentence...
    for text in train_samples_text+val_samples_text:
        
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_id = tokenizer(text, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_id['input_ids']))

    logger.info('Max text length: ' + str(max_len))
    
    batch_size = args.batch_size

    (train_input_ids, train_attention_masks, train_samples_label_tensor) = make_smart_batches(train_samples_text, train_samples_label, batch_size, logger, tokenizer, max_len)
    (val_input_ids, val_attention_masks, val_samples_label_tensor) = make_smart_batches(val_samples_text, val_samples_label, batch_size, logger, tokenizer, max_len)

    logger.info('{:>5,} training samples'.format(train_size))
    logger.info('{:>5,} validation samples'.format(val_size))

    model = BertForSequenceClassification.from_pretrained(        
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = no_of_labels, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.        
    )
    
    
    model = model.to(device=device)    
    # model.cuda(device=device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = hparams["adam_epsilon"] # args.adam_epsilon  - default is 1e-8.
                    )
    epochs = 4

    total_steps = len(train_input_ids) * epochs

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

        step = 0
        for batch in zip(train_input_ids, train_attention_masks, train_samples_label_tensor):
        
            if step % 40 == 0 and not step == 0:               
                logger.info('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_input_ids)))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)           

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            total_train_loss += loss.detach().cpu().numpy()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            step+=1
        
        avg_train_loss = total_train_loss / len(train_input_ids)            

        logger.info("")
        logger.info("Average training loss: {0:.2f}".format(avg_train_loss))

            
        logger.info("")
        logger.info("Running Validation...")

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in zip(val_input_ids, val_attention_masks, val_samples_label_tensor):
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
            

        avg_val_accuracy = total_eval_accuracy / len(val_input_ids)
        logger.info("Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(val_input_ids)
               
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
                    required=False,
                    help="Possible values: head-only, tail-only, and head-and-tail")
    parser.add_argument("--dataset_amount",
                    default="full",
                    type=str,
                    required=False,
                    help="Accepted values - 'same size' or 'full'")
    parser.add_argument("--binary_classifier",
                    # type=bool,
                    dest='binary_classifier',
                    action='store_true',
                    help="")
    parser.add_argument("--no_binary_classifier",
                    # type=bool,
                    dest='binary_classifier',
                    action='store_false',
                    help="")
    parser.add_argument("--data_1",
                    default=0,
                    type=int,
                    required=False,
                    help="left-0, center-1, right-2")
    parser.add_argument("--data_2",
                    default=2,
                    type=int,
                    required=False,
                    help="left-0, center-1, right-2")
    parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    required=False,
                    help="")
    parser.add_argument("--batch_size",
                    default=None,
                    type=int,
                    required=True,
                    help="")
    
    parser.set_defaults(binary_classifier=True)



    args = parser.parse_args()

    hyperparams = [       
        {
            # "learning_rate": 2e-5,
            # "batch_size": 8,
            "seed_val": 23,
            "adam_epsilon": 1e-8
        },  
    ]

    for hparams in hyperparams:         
        # myprint(f"args: {args}")    
        # myprint(f"hparams: {hparams}")
        train_model(args, hparams)
        

    