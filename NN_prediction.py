import pandas as pd
import numpy as np

from cleaning_for_GUI import text_preprocessing

import torch

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler




def neural_net_predictor(task, pre_trained_model, text):
    """
    Function that, given a task, pre-trained model and text, will return sentiment using .predict method

    Parameters:
    ---
    task: string
            Possible Inputs: Emotion Identification, Positive/Neutral/Negative, Star Rating

    pre_trained_model: string
            Possible Inputs:

    text: string
            Text meant for sentiment prediction
    """


    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not the cpu will be used
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    def preprocessing_for_bert(data):
                """Perform required preprocessing steps for pretrained BERT.
                @param    data (np.array): Array of texts to be processed.
                @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
                @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                            tokens should be attended to by the model.
                """
                # Create empty lists to store outputs
                input_ids = []
                attention_masks = []

                # For every sentence...
                for sent in data:
                    # `encode_plus` will:
                    #    (1) Tokenize the sentence
                    #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
                    #    (3) Truncate/Pad sentence to max length
                    #    (4) Map tokens to their IDs
                    #    (5) Create attention mask
                    #    (6) Return a dictionary of outputs
                    encoded_sent = tokenizer.encode_plus(
                        text=text_preprocessing(sent),  # Preprocess sentence
                        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                        max_length=512,                  # Max length to truncate/pad
                        truncation=True,
                        pad_to_max_length=True,         # Pad sentence to max length
                        #return_tensors='pt',           # Return PyTorch tensor
                        return_attention_mask=True      # Return attention mask
                        )
                    
                    # Add the outputs to the lists
                    input_ids.append(encoded_sent.get('input_ids'))
                    attention_masks.append(encoded_sent.get('attention_mask'))

                # Convert lists to tensors
                input_ids = torch.tensor(input_ids)
                attention_masks = torch.tensor(attention_masks)

                return input_ids, attention_masks



    def bert_predict(model, test_dataloader):
            import torch.nn.functional as F
            """Perform a forward pass on the trained BERT model to predict probabilities
            on the test set.
            """
            # Put the model into the evaluation mode. The dropout layers are disabled during
            # the test time.
            model.eval()

            all_logits = []

            # For each batch in our test set...
            for batch in test_dataloader:
                # Load batch to GPU
                b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

                # Compute logits
                with torch.no_grad():
                    logits = model(b_input_ids, b_attn_mask)
                all_logits.append(logits)
            
            # Concatenate logits from each batch
            all_logits = torch.cat(all_logits, dim=0)

            # Apply softmax to calculate probabilities
            probs = F.softmax(all_logits, dim=1).cpu().numpy()

            return probs



    my_string = [f'{text}']
    abcdefg = {'text' : my_string}
    test_words = pd.DataFrame(data=abcdefg)

    test_inputs, test_masks = preprocessing_for_bert(test_words.text)
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)



    probs = bert_predict(pre_trained_model, test_dataloader)
    preds = np.argmax(probs, axis=1)

    predicted_sentiments = []
    if task == "Emotion Identification":
        for pred in preds:
            if pred == 0:
                predicted_sentiment = 'Sadness'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==1:
                predicted_sentiment = 'Joy'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==2:
                predicted_sentiment = 'Love'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==3:
                predicted_sentiment = 'Anger'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==4:
                predicted_sentiment = 'Fear'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==5:
                predicted_sentiment = 'Surprise'
                predicted_sentiments.append(predicted_sentiment)

    if task == "Positive/Neutral/Negative":
        for pred in preds:
            if pred == 0:
                predicted_sentiment = 'Negative'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==1:
                predicted_sentiment = 'Positive'
                predicted_sentiments.append(predicted_sentiment)
            elif pred ==2:
                predicted_sentiment = 'Neutral'
                predicted_sentiments.append(predicted_sentiment)
    if task == "Star Rating":
        for pred in preds:
            predicted_sentiment = pred+1
            predicted_sentiments.append(predicted_sentiment)
    return predicted_sentiments[0]