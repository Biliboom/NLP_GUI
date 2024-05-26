import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from typing import List

from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def random_forest_predictor(task, pre_trained_model, text):
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

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


    def get_embeddings(texts: List[str], batch_size: int):
        all_embeddings = []
        print(f"Total number of records: {len(texts)}")
        print(f"Num batches: {(len(texts) // batch_size) + 1}")

        # Extract embeddings for the texts in batches
        for start_index in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[start_index:start_index + batch_size]

            # Generate tokens and move input tensors to GPU
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Extract the embeddings. no_grad because the gradient does not need to be computed
            # since this is not a learning task
            with torch.no_grad():
                outputs = model(**inputs)

            # Get the last hidden stated and pool them into a mean vector calculated across the sequence length dimension
            # This will reduce the output vector from [batch_size, sequence_length, hidden_layer_size]
            # to [batch_size, hidden_layer_size] thereby generating the embeddings for all the sequences in the batch
            last_hidden_states = outputs.last_hidden_state
            embeddings = torch.mean(last_hidden_states, dim=1).cpu().tolist()

            # Append to the embeddings list
            all_embeddings.extend(embeddings)

        return all_embeddings

    d = [f"{text}"]
    text_embeddings = get_embeddings(texts=d, batch_size=256)
    text_embeddings_df = pd.DataFrame(text_embeddings)
    y_pred = pre_trained_model.predict(text_embeddings_df)

    predicted_sentiments=[]
    if task == "Emotion Identification":
        if y_pred == 0:
            predicted_sentiment = 'Sadness'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==1:
            predicted_sentiment = 'Joy'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==2:
            predicted_sentiment = 'Love'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==3:
            predicted_sentiment = 'Anger'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==4:
            predicted_sentiment = 'Fear'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==5:
            predicted_sentiment = 'Surprise'
            predicted_sentiments.append(predicted_sentiment)

    if task == "Positive/Neutral/Negative":
        if y_pred == 0:
            predicted_sentiment = 'Negative'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==1:
            predicted_sentiment = 'Positive'
            predicted_sentiments.append(predicted_sentiment)
        elif y_pred ==2:
            predicted_sentiment = 'Neutral'
            predicted_sentiments.append(predicted_sentiment)
    
    return predicted_sentiments[0]






