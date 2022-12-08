# Using a pretrained BERT model to classify the sentiment of a given tweet.
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn
import re

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

DATA_PATH = "src/data/"


def clean_tweet(tweet):
    tweet = tweet.lower()
    # detect and remove (hash)tags
    tweet = re.sub(r"@[A-Za-z0-9_?-]+", "usertag", tweet)
    tweet = re.sub(r"#[A-Za-z0-9_?-]+", "trendtag", tweet)
    # detect and remove urls
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", "", tweet)

    # remove all non alphanumeric characters
    tweet = re.sub(r"[^a-zA-Z0-9,]", " ", tweet)

    tweet = re.sub(r" +", " ", tweet)
    # strip trailing and leading whitespaces
    tweet = tweet.strip()
    return tweet


def preprocess_dataset(filename: str) -> pd.DataFrame:
    set_df = pd.read_csv(DATA_PATH + filename, encoding="latin-1", header=None)
    set_df.drop(columns=[1, 2, 3, 4], axis=1, inplace=True)
    set_df.columns = ["sentiment", "text"]

    # Replace 4 with 1 since the dataset does not contain any sentiments between 0 and 4
    set_df.sentiment.replace(4, 1, inplace=True)
    set_df["text"] = set_df.text.apply(clean_tweet)
    
    return set_df


def tokenize_dataset(dataset: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []
    i = 0
    for tweet in dataset.OriginalTweet.values:
        if i % 1000 == 0:
            print(f"Tokenizing tweet {i}")
        i += 1
        encoded_dict = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(dataset.Sentiment.apply(sentiment_to_int).values)

    return input_ids, attention_masks, labels


def sentiment_to_int(sentiment: str) -> int:
    if sentiment == "Extremely Negative":
        return 0
    elif sentiment == "Negative":
        return 1
    elif sentiment == "Neutral":
        return 2
    elif sentiment == "Positive":
        return 3
    elif sentiment == "Extremely Positive":
        return 4
    else:
        raise ValueError(f"Sentiment {sentiment} is not valid")


def train_model(train_df: pd.DataFrame):
    input_ids, attention_masks, labels = tokenize_dataset(train_df)
    data = TensorDataset(input_ids, attention_masks, labels)
    training_data, validation_data = random_split(data, [40000, 1157])
    training_dataloader = DataLoader(training_data, sampler=RandomSampler(training_data), batch_size=16)
    validation_dataloader = DataLoader(validation_data, sampler=RandomSampler(validation_data), batch_size=16)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5, output_attentions=False, output_hidden_states=False)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(training_dataloader) * epochs)
    
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(training_dataloader):
            if step % 5 == 0 and not step == 0:
                print(f"Batch {step} of {len(training_dataloader)}")
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(training_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        evaluate_model(validation_dataloader, model)


def evaluate_model(validation_dataloader: DataLoader, model: BertForSequenceClassification):
    model.eval()
    total_eval_loss = 0
    preds, truth = [], []
    for step, batch in enumerate(validation_dataloader):
        batch = tuple(t.cuda() for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels = b_labels.to('cpu').numpy()
        preds.append(logits)
        truth.append(labels)
    preds = np.argmax(preds, axis=0).flatten()
    truth = truth.flatten()
    accuracy = np.sum(preds=truth) / len(truth)
    print(f"Accuracy: {accuracy}")
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print(f"Validation loss: {avg_val_loss}")



def main():
    train_df = pd.read_csv(DATA_PATH + "coronanlp/Corona_NLP_train.csv", encoding="latin-1", header=0)
    train_df.columns = ["UserName", "ScreenName", "Location", "TweetAt", "OriginalTweet", "Sentiment"]
    train_df.drop(columns=["UserName", "ScreenName", "Location", "TweetAt"], axis=1, inplace=True)
    train_df.OriginalTweet = train_df.OriginalTweet.apply(clean_tweet)

    train_model(train_df)

    # test_df = pd.read_csv(DATA_PATH + "coronanlp/Corona_NLP_test.csv", encoding="latin-1", header=0)
    # test_df.columns = ["UserName", "ScreenName", "Location", "TweetAt", "OriginalTweet", "Sentiment"]
    # test_df.drop(columns=["UserName", "ScreenName", "Location", "TweetAt"], axis=1, inplace=True)
    # test_df.OriginalTweet.apply(clean_tweet, inplace=True)

    



if __name__ == "__main__":
    main()


