import pandas as pd
import numpy as np

import torch                     #importing pytorch
import torch.nn as nn            #While this project uses pytorch for all ML purposes, one could also use tensorflow

from transformers import BertTokenizer  
from transformers import BertModel

from cleaning import convert      #function that turns data file into pandas dataframe
from cleaning import splitter     #function for splitting data into test,train and validation data:     returns  X_test, y_test, X_train, y_train, X_val, y_val
from cleaning import text_preprocessing

import matplotlib.pyplot as plt   #These two packages are used to visualize the evaluation of our ML models
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

import dill as pickle   #We import dill instead of pickle here as pickle doesn't function as inteded for NN



def bert_nn(filename,size, number_of_epochs = 2, test=False):
    root = f"C:/Users/Bijan-PC/Documents/Coding/UNIL/Data Analysis/ADA_Project/ADA_Final/dat_cleaned/{size}"      #base location for where all data files are
    df = convert(root,filename)            #Definining dataframe where all training and validation data will be pulled from
    number_of_labels = len(df["label"].value_counts())          #number of labels (will be used when training)
    
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not the cpu will be used
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Defining train-validation-test split (90-10-10)    
    X = df.text.values
    y = df.label.values

    X_test, y_test, X_train, y_train, X_val, y_val = splitter(X,y)

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Encode our concatenated data
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in X]

    # Define the maximum length
    max_len = max([len(sent) for sent in encoded_tweets])
    if max_len < 513:
        MAX_LEN = max_len
    else:
        MAX_LEN = 512                      
    print('Max length: ', MAX_LEN)


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
                max_length=MAX_LEN,                  # Max length to truncate/pad
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

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)


    #Now let’s tokenize our data.
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = 32

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



    # Create the BertClassfier class
    class BertClassifier(nn.Module):
        """Bert Model for Classification Tasks.
        """
        def __init__(self, freeze_bert=False):
            """
            @param    bert: a BertModel object
            @param    classifier: a torch.nn.Module classifier
            @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
            """
            super(BertClassifier, self).__init__()
            # Specify hidden size of BERT, hidden size of our classifier, and number of labels
            D_in, H, D_out = 768, 50, number_of_labels

            # Instantiate BERT model
            self.bert = BertModel.from_pretrained('bert-base-uncased')

            # Instantiate an one-layer feed-forward classifier
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(H, D_out)
            )

            # Freeze the BERT model
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
        def forward(self, input_ids, attention_mask):
            """
            Feed input to BERT and the classifier to compute logits.
            @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                        max_length)
            @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                        information with shape (batch_size, max_length)
            @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                        num_labels)
            """
            # Feed input to BERT
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            
            # Extract the last hidden state of the token `[CLS]` for classification task
            last_hidden_state_cls = outputs[0][:, 0, :]

            # Feed input to classifier to compute logits
            logits = self.classifier(last_hidden_state_cls)

            return logits


    from transformers import AdamW, get_linear_schedule_with_warmup

    def initialize_model(epochs=4):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        bert_classifier = BertClassifier(freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        bert_classifier.to(device)

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
        return bert_classifier, optimizer, scheduler




    import random
    import time

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    def set_seed(seed_value=42):
        """Set seed for reproducibility.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
        """Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()
                
            

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-"*70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, class_report, conf_matirx = evaluate(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-"*70)

                
            print("\n")

        print(class_report)
        
        print(conf_matirx)

        print("Training complete!")

    def plot_confusion_matrix(y_preds, y_true, labels=None):                                #Defining function that plots results onto a confusion matrix
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(number_of_labels, number_of_labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False) 
        plt.title("Normalized confusion matrix")
        #plt.show()

    def evaluate(model, val_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []
        val_preds = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            
            #Get the predictions which will be used for classification report
            confusion_preds = preds.cpu().numpy()
            val_preds.append(confusion_preds)

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        #Display classification report
        val_preds = np.concatenate(val_preds)
        class_report = classification_report(y_val, val_preds)
        #cm = confusion_matrix(y_val, val_preds) 

        #Display confusion matrix
        cm = plot_confusion_matrix(val_preds, y_val)

        return val_loss, val_accuracy, class_report, cm


    set_seed(42)    # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = initialize_model(epochs= number_of_epochs)                          #Initialize the model
    train(bert_classifier, train_dataloader, val_dataloader, epochs= number_of_epochs, evaluation=True)         #Train the model
    savename = filename.replace(".csv","")                                #Create variable that is used for naming the saved version (pickle) of the model, which will be used in the GUI
    with open(f'Pickle/Neural Net/{savename}.pickle', 'wb') as handle:
        pickle.dump(bert_classifier, handle)                           #Save the model to Pickle folder


    def tester():                                                                           #Function that evaluates the performance of the model on test data
        def plot_confusion_matrix(y_preds, y_true, labels=None):
            cm = confusion_matrix(y_true, y_preds, normalize="true")
            fig, ax = plt.subplots(figsize=(number_of_labels, number_of_labels))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
            disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False) 
            plt.title("Normalized confusion matrix")
            #plt.show()
        

        import torch.nn.functional as F
        def bert_predict(model, test_dataloader):
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
        
        print("-"*70)
        print("Evaluating on Test Data")
        test_df = {'label': y_test, 'text' : X_test}
        test_data = pd.DataFrame(data=test_df)

        # Run `preprocessing_for_bert` on the test set
        print('Tokenizing data...')
        test_inputs, test_masks = preprocessing_for_bert(test_data.text)

        # Create the DataLoader for our test set
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

        # Compute predicted probabilities on the test set
        probs = bert_predict(bert_classifier, test_dataloader)

        # Get predictions from the probabilities

        #Threshold (only useful for binary classification where getting false negative, for example, would be detrimental)
        threshold = 0.95

        #returns numpy array with the index of the highest probability for each prediction
        preds = np.argmax(probs, axis=1)

        #returns f1 score for test data, comparing the predicted emotions with the real labeled emotions
        
        return classification_report(y_test, preds), plot_confusion_matrix(y_test, preds, labels=None)
        
    if test == True:
        print(tester())
        