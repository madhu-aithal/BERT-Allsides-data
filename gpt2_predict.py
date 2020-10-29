import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import logging
logging.basicConfig(level=logging.INFO)


def predict_headline_given_description(news_desc, model, tokenizer):

    # Encode a text inputs
    text = "<BOS> "+news_desc + " <SEP> "
    text = text.replace("\n", " ").strip()
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict all tokens
    last_token = ""
    headline = ""
    predicted_text = ""

    with torch.no_grad():
        while last_token != "<EOS>":
            outputs = model(tokens_tensor)
            predictions = outputs[0]

            # get the predicted next sub-word (in our case, the word 'man')
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

            # print(predicted_text.split()[-1])
            last_token = predicted_text.split()[-1].strip()
            headline += last_token+" "

            indexed_tokens = tokenizer.encode(predicted_text)

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to('cuda')
        
        headline = predicted_text.split("<SEP>", 1)[1].strip()
        print(text)
        print()
        print(headline)


def predict_LCR_headline_given_allsides_desc(text, headline_label, model, tokenizer):        
    # Encode a text inputs
    text = "<BOS> "+text + " <SEP> " + headline_label
    text = text.replace("\n", " ").strip()
    
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict all tokens
    last_token = ""
    headline = ""
    predicted_text = ""

    with torch.no_grad():
        while last_token != "<EOS>":
            outputs = model(tokens_tensor)
            predictions = outputs[0]

            # get the predicted next sub-word (in our case, the word 'man')
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

            # print(predicted_text.split()[-1])
            last_token = predicted_text.split()[-1].strip()
            # print(last_token)
            headline += last_token+" "

            indexed_tokens = tokenizer.encode(predicted_text)

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to('cuda')
        
        headline = predicted_text.split("<SEP>", 1)[1].strip()
        print(text)
        print()
        print(headline)

# input - allsides_desc <SEP> news_desc <SEP> <LEFT/CENTER/RIGHT> --> headline
def predict_LCR_headline_given_allsides_desc_news_desc(allsides_desc, news_desc, headline_label, model, tokenizer):        
    # Encode a text inputs
    text = "<BOS> "+allsides_desc + " <SEP> " + news_desc + " <SEP> <" + headline_label.upper() + ">"

    text = text.replace("\n", " ").strip()
    
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    # Predict all tokens
    last_token = ""
    headline = ""
    predicted_text = ""

    with torch.no_grad():
        while last_token != "<EOS>":
            outputs = model(tokens_tensor)
            predictions = outputs[0]

            # get the predicted next sub-word (in our case, the word 'man')
            predicted_index = torch.argmax(predictions[0, -1, :]).item()
            predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

            # print(predicted_text.split()[-1])
            last_token = predicted_text.split()[-1].strip()
            # print(last_token)
            headline += last_token+" "

            indexed_tokens = tokenizer.encode(predicted_text)

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to('cuda')
        
        headline = predicted_text.split("<"+headline_label+">")[1].strip()
        print(text)
        print()
        print(headline)

if __name__ == "__main__":
    text = '''The Senate Judiciary Committee voted 12-0 to confirm Judge Amy Coney Barrett to the Supreme Court despite the Democratic members of the panel boycotting the hearing.

The nomination will now move to a full Senate vote on Monday.

Sen. Lindsey Graham on Thursday said he will not allow Democrats to “take over” the Senate Judiciary Committee’s confirmation vote for Barrett after they made good on their threat to boycott the hearing.

“We’re not going to allow them to take over the committee,” Graham said in his opening statement. “They made...'''

    # text = text.replace("\n", " ").strip()

    # Generate news headline given 'news_desc'
    # tokenizer = GPT2Tokenizer.from_pretrained('output_gpt2_given_news_desc_predict_headline')
    # model = GPT2LMHeadModel.from_pretrained('output_gpt2_given_news_desc_predict_headline')

    # predict_headline_given_description(text, model, tokenizer)


    # Generate news headline given allsides_desc <ideology>
    # tokenizer = GPT2Tokenizer.from_pretrained('output_gpt2_allsides_desc_headline')
    # model = GPT2LMHeadModel.from_pretrained('output_gpt2_allsides_desc_headline')

    # headline_label = "<LEFT>"
    # predict_LCR_headline_given_allsides_desc(text, headline_label, model, tokenizer)

    # headline_label = "<CENTER>"
    # predict_LCR_headline_given_allsides_desc(text, headline_label, model, tokenizer)

    # headline_label = "<RIGHT>"
    # predict_LCR_headline_given_allsides_desc(text, headline_label, model, tokenizer)

    

    allsides_desc = '''
    Republicans on the Senate Judiciary Committee voted 12-0 to advance Amy Coney Barrett’s Supreme Court nomination to the full Senate. The committee’s ten Democratic members boycotted the vote by refusing to attend the committee hearing. Committee Chairman Lindsey Graham moved forward without the Democrats, despite committee rules requiring at least two members of the minority to be present for a vote. Graham instead followed broader Senate rules only requiring a simple majority of committee members for a vote. A full Senate vote on Barrett’s nomination is expected Monday.
    Coverage in left-rated outlets was more likely to refer to the boycott as “symbolic” and to include that no Supreme Court nominee has ever been confirmed this close to a presidential election. Coverage in right-rated outlets tended to say that the committee voted “unanimously” to move forward with confirmation.'''

    news_desc = '''
    Republicans on the Senate Judiciary Committee voted on Thursday to advance Judge Amy Coney Barrett’s Supreme Court nomination after Democrats boycotted the vote.

The panel voted 12-0 to send Barrett’s nomination to the full Senate, paving the way for President Trump’s nominee to be confirmed to the Supreme Court early next week. Every Republican on the panel supported her nomination and no Democratic senator voted.

Every GOP senator was present for the vote, meeting the committee's rule that 12 members of the panel must be present to report a...'''

    tokenizer = GPT2Tokenizer.from_pretrained('output_gpt2_allsides_desc_news_desc_headline')
    model = GPT2LMHeadModel.from_pretrained('output_gpt2_allsides_desc_news_desc_headline')
    
    headline_label = "CENTER"
    predict_LCR_headline_given_allsides_desc_news_desc(allsides_desc, news_desc, headline_label, model, tokenizer)