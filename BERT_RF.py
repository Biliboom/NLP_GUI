import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cleaning import convert
from cleaning import splitter
from cleaning import text_preprocessing

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List
import torch

import pickle



def bert_rf(filename, size, test=False):

    root = f"C:/Users/Bijan-PC/Documents/Coding/UNIL/Data Analysis/ADA_Project/ADA_Final/dat_cleaned/{size}"      #base location for where all data files are
    df = convert(root,filename)            #Definining dataframe where all training and validation data will be pulled from
    number_of_labels = len(df["label"].value_counts())          #number of labels (will be used when training)
    
    def train_model(data: pd.DataFrame, labels: pd.Series):

        # Initialize the Random Forest Classifier
        clf = RandomForestClassifier(random_state=3137)

        # Define hyperparameters and values to tune
        param_grid = {
            'max_depth': [3, 5, 7, 9],
        }

        print(f"Number of rows in training data: {len(data)}")

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring="roc_auc_ovo",
                                cv=5, verbose=3)
        grid_search.fit(data, labels)

        # Get the best hyperparameters
        best_max_depth = grid_search.best_params_['max_depth']

        final_clf = RandomForestClassifier(
                                        max_depth=best_max_depth,
                                        random_state=3137)
        final_clf.fit(data, labels)

        return final_clf

    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # For Sklearn Random Forest, using gpu is not available so we will have to run it on cpu
    device = torch.device("cpu")
    model = model.to(device)

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
    


    # Split into train and test
    X = df.text.values
    y = df.label.values

    X_test, y_test, X_train, y_train, X_val, y_val = splitter(X,y)

    X_train = pd.DataFrame(X_train, columns=['text'])
    X_val = pd.DataFrame(X_val, columns=['text'])
    X_test = pd.DataFrame(X_test, columns=['text'])

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    y_val = pd.Series(y_val)

    for dataset in [X_test, X_train, X_val]:
        dataset["text_cleaned"] = dataset["text"].apply(lambda x: text_preprocessing(x))
        print(f'Cleaned {len(dataset["text_cleaned"])} records in dataset')

    # Get embeddings for the training and test set
    train_embeddings = get_embeddings(texts=X_train["text_cleaned"].tolist(), batch_size=256)
    train_embeddings_df = pd.DataFrame(train_embeddings)

    test_embeddings = get_embeddings(texts=X_test["text_cleaned"].tolist(), batch_size=256)
    test_embeddings_df = pd.DataFrame(test_embeddings)

    val_embeddings = get_embeddings(texts=X_val["text_cleaned"].tolist(), batch_size=256)
    val_embeddings_df = pd.DataFrame(val_embeddings)



    from sklearn.model_selection import GridSearchCV

    def plot_confusion_matrix(y_preds, y_true, labels=None):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(number_of_labels, number_of_labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False) 
        plt.title("Normalized confusion matrix")
        #plt.show()

    # Train model
    print("Training...")
    rf_model = train_model(data=train_embeddings_df, labels=y_train)
    savename = filename.replace(".csv","")                                #Create variable that is used for naming the saved version (pickle) of the model, which will be used in the GUI
    with open(f"Pickle/Random Forest/{savename}.pickle", 'wb') as handle:
        pickle.dump(rf_model, handle)     


    # Predict from model on validation data
    y_pred_val = rf_model.predict(val_embeddings_df)
    y_pred_val = pd.Series(y_pred_val)

    # Evaluate model (validation data)
    print("-"*70)
    print("Evaluation on Validation Data")
    print(f"Classification report:\n{classification_report(y_val, y_pred_val)}")
    print(plot_confusion_matrix(y_val, y_pred_val))

    if test == True:
            # Predict from model on test data
        y_pred_test = rf_model.predict(test_embeddings_df)
        y_pred_test = pd.Series(y_pred_test)

            # Evaluate model (test data)
        print("-"*70)
        print("Evaluation on Test Data")
        print(f"Classification report:\n{classification_report(y_test, y_pred_test)}")
        print(plot_confusion_matrix(y_test, y_pred_test))








