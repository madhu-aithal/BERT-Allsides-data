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
import math
from torch.utils.tensorboard import SummaryWriter

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_model(args: dict, hparams:dict):

    
    
    file = args.dataset_filepath
    # truncation = args.truncation

    seed_val = hparams["seed_val"]
    device = utils.get_device(device_no=args.device_no)
    saves_dir = "saves/bert/"

    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    time = datetime.datetime.now()

    saves_path = os.path.join(saves_dir, utils.get_filename(time))
    Path(saves_path).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(saves_path, "training.log")

    summary_filename = os.path.join(saves_path, "tensorboard_summary")
    writer = SummaryWriter(summary_filename)

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

    samples = utils.read_samples(file)

    article_type_map = {}

    if not args.nonpair_data:        
        if args.lcr:
            samples = [[val[0].lower()+" [SEP] "+val[1].lower()+" [SEP] "+val[2].lower(), val[3], val[4]] for val in samples]        
        else:
            samples = [[val[0].lower()+" [SEP] "+val[1].lower(), val[2], val[3]] for val in samples]

    if args.group_by_domestic:
        samples_new = []
        for s in samples:
            # article_type_map[s[0]] = s[2]
            if s[2] == "domestic":
                samples_new.append([s[0], s[1], 0])
            else:
                samples_new.append([s[0], s[1], 1])

        samples = samples_new
    # samples = samples[:100]
    # if args.binary_classifier:        
    #     samples = utils.read_pairwise(file, args.data_1, args.data_2, dataset_amount=args.dataset_amount)
    # else:
    #     samples = utils.read_and_sample(file, dataset_amount=args.dataset_amount)

    no_of_labels = len(set([val[1] for val in samples]))

    logger.info("No of unique labels: "+str(no_of_labels))

    # train_size = int(0.9 * len(samples))
    # val_size = len(samples) - train_size

    # random.shuffle(samples)

    # train_samples = samples[:train_size]
    # val_samples = samples[train_size:]
    
    # train_samples_text = [val[0] for val in train_samples]
    # train_samples_label = [val[1] for val in train_samples]
    # val_samples_text = [val[0] for val in val_samples]
    # val_samples_label = [val[1] for val in val_samples]

    samples_text = [val[0] for val in samples]
    samples_label = [val[1] for val in samples]
    if args.group_by_domestic:
        samples_article_type = [val[2] for val in samples]

    max_len = 0

    input_ids = []
    attention_masks = []    

    # For every sentence...
    for text in samples_text:                
        input_id = tokenizer(text, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_id['input_ids']))

    logger.info('Max text length: ' + str(max_len))

    max_len = pow(2, math.ceil(math.log2(max_len)))
    max_len = min(512, max_len)

    for text in samples_text:
        input_id = tokenizer(text, add_special_tokens=True)
        if len(input_id) > 512:            
            if args.truncation == "tail-only":
                input_id['input_ids'] = [tokenizer.cls_token_id]+input_id['input_ids'][-511:]      
            elif args.truncation == "head-and-tail":
                input_id['input_ids'] = [tokenizer.cls_token_id]+input_id['input_ids'][1:129]+input_id['input_ids'][-382:]+[tokenizer.sep_token_id]
            else:
                input_id['input_ids'] = input_id['input_ids'][:511]+[tokenizer.sep_token_id]
                
            input_ids.append(torch.tensor(input_id['input_ids']).view(1,-1))
            attention_masks.append(torch.ones([1,len(input_id['input_ids'])], dtype=torch.long))
        else:
            encoded_dict = tokenizer(
                                text,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )            
            input_ids.append(encoded_dict['input_ids'])                        
            attention_masks.append(encoded_dict['attention_mask'])
    
    batch_size = args.batch_size
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(samples_label)
    if args.group_by_domestic:
        samples_article_type_tensor = torch.tensor(samples_article_type)

    # Combine the training inputs into a TensorDataset.
    if args.group_by_domestic:
        dataset = TensorDataset(input_ids, attention_masks, labels, samples_article_type_tensor)
    else:
        dataset = TensorDataset(input_ids, attention_masks, labels)

    # (train_input_ids, train_attention_masks, train_samples_label_tensor) = make_smart_batches(train_samples_text, train_samples_label, batch_size, logger, tokenizer, max_len)
    # (val_input_ids, val_attention_masks, val_samples_label_tensor) = make_smart_batches(val_samples_text, val_samples_label, batch_size, logger, tokenizer, max_len)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info('{:>5,} training samples'.format(train_size))
    logger.info('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

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
    epochs = args.n_epochs

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    best_stats = {
        'epoch': 0,
        'training_loss': -sys.maxsize,
        'training_accuracy': -sys.maxsize,
        'validation_loss': -sys.maxsize,
        'validation_accuracy': -sys.maxsize,
    }
    
    for epoch_i in range(0, epochs):

        if len(training_stats) > 2:
            if training_stats[-1]['validation_accuracy'] <= training_stats[-2]['validation_accuracy'] \
                and training_stats[-2]['validation_accuracy'] <= training_stats[-3]['validation_accuracy']:
                break

        correct_counts = {
            "domestic": 0,
            "international": 0
        }
        total_counts = {
            "domestic": 0,
            "international": 0
        }
        
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')
        
        total_train_loss = 0
        total_train_accuracy = 0

        model.train()

        step = 0
        for step, batch in enumerate(train_dataloader):
        
            if step % 40 == 0 and not step == 0:               
                logger.info('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

            b_input_ids = batch[0].to(device=device)
            b_input_mask = batch[1].to(device=device)
            b_labels = batch[2].to(device=device) 
            
            # Converting labels to float32 because I was getting some runtime error. 
            # Not sure why we need to make labels float32. Keeping it Long or int64 works in case of headlines.
            # b_labels = batch[2].to(device=device, dtype=torch.float32) 

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += flat_accuracy(logits, label_ids)
            total_train_loss += loss.detach().cpu().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step+=1

        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        logger.info("")
        logger.info("Average training accuracy: {0:.2f}".format(avg_train_accuracy))
        
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
            if args.group_by_domestic:
                b_article_types = batch[3].to(device)
            
            with torch.no_grad():        

                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            total_eval_loss += loss.detach().cpu().numpy()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

            if args.group_by_domestic:   
                for idx in range(len(b_labels)):
                    pred = np.argmax(logits[idx]) == label_ids[idx]
                    if b_article_types[idx] == 0:
                        if pred == True:
                            correct_counts["domestic"] += 1
                        total_counts["domestic"] += 1
                    

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logger.info("Avg validation accuracy: {0:.2f}".format(avg_val_accuracy))

        if args.group_by_domestic:
            avg_val_accuracy_domestic = correct_counts["domestic"]/total_counts["domestic"]

            logger.info("Domestic validation accuracy: {0:.2f}".format(avg_val_accuracy_domestic))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
               
        logger.info("Validation Loss: {0:.2f}".format(avg_val_loss))        

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'training_loss': avg_train_loss,
                'training_accuracy': avg_train_accuracy,
                'validation_loss': avg_val_loss,
                'validation_accuracy': avg_val_accuracy,
            }
        )
        if avg_val_accuracy > best_stats['validation_accuracy']:
            best_stats = {
                'epoch': epoch_i + 1,
                'training_loss': avg_train_loss,
                'training_accuracy': avg_train_accuracy,
                'validation_loss': avg_val_loss,
                'validation_accuracy': avg_val_accuracy,
            }

        writer.add_scalars('losses_and_accuracies', {
            'training_loss': avg_train_loss,
            'training_accuracy': avg_train_accuracy,
            'validation_loss': avg_val_loss,
            'validation_accuracy': avg_val_accuracy,
        }, epoch_i+1)

        model_save_path = os.path.join(saves_path, "model_"+str(epoch_i+1)+"epochs")
        torch.save(model, model_save_path)        

    logger.info("")
    logger.info("Training complete!")
    logger.info("Best stats")     
    logger.info("training_accuracy: {}".format(best_stats['training_loss']))
    logger.info("training_loss: {}".format(best_stats['training_loss']))
    logger.info("validation_accuracy: {}".format(best_stats['validation_accuracy']))
    logger.info("validation_loss: {}".format(best_stats['validation_loss']))

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
    # parser.add_argument("--binary_classifier",
    #                 # type=bool,
    #                 dest='binary_classifier',
    #                 action='store_true',
    #                 help="")
    # parser.add_argument("--no_binary_classifier",
    #                 # type=bool,
    #                 dest='binary_classifier',
    #                 action='store_false',
    #                 help="")
    parser.add_argument("--LCR",
                    # type=bool,
                    dest='lcr',
                    action='store_true',
                    help="")
    parser.add_argument("--no_LCR",
                    # type=bool,
                    dest='lcr',
                    action='store_false',
                    help="")
    parser.add_argument("--nonpair_data",
                    # type=bool,
                    dest='nonpair_data',
                    action='store_true',
                    help="")    
    parser.add_argument("--group_by_domestic",
                    # type=bool,
                    dest='group_by_domestic',
                    action='store_true',
                    help="")
    # parser.add_argument("--data_1",
    #                 default=0,
    #                 type=int,
    #                 required=False,
    #                 help="left-0, center-1, right-2")
    # parser.add_argument("--data_2",
    #                 default=2,
    #                 type=int,
    #                 required=False,
    #                 help="left-0, center-1, right-2")
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
    parser.add_argument("--n_epochs",
                    default=4,
                    type=int,
                    required=False,
                    help="")
    
    # parser.set_defaults(binary_classifier=True)
    parser.set_defaults(lcr=False)
    parser.set_defaults(nonpair_data=False)
    parser.set_defaults(group_by_domestic=False)
    
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
        
'''
How to run:
1. nohup python train.py --dataset_filepath \
    "/data/madhu/allsides_scraped_data/new_data_sept_25/processed_data/left_labeled_article_headline_LR.pickle" \
    --device_no 1 --batch_size 64 --n_epochs 12 --LCR &> nohup_left_labeled_article_headline_LR.out &

New command - Oct 29
2. nohup python train_bert.py --dataset_filepath 
/data/madhu/allsides_scraped_data/new_data_oct_7/processed_data_with_article_type/left_labeled_article_headline_LR.pickle 
--device_no 1 --batch_size 64 --n_epochs 16 --no_LCR &> nohup_left_labeled_headline_LR.out &

'''