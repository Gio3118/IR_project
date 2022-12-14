import os
import torch
import pandas as pd
import re
import time
import numpy as np
from datetime import datetime as dt

from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.optim import SGD, AdamW, lr_scheduler


SENTIMENTS = {
    "extremely negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "extremely positive": 4,
}


def sentiment_to_int(sentiment: str) -> int:
    return SENTIMENTS.get(sentiment.lower(), None)


def clean_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    # detect and remove (hash)tags
    tweet = re.sub(r"@[A-Za-z0-9_?-]+", "usertag", tweet)
    tweet = re.sub(r"#[A-Za-z0-9_?-]+", "trendtag", tweet)

    # detect and remove urls
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", "", tweet)

    # remove all non alphanumeric characters
    tweet = re.sub(r"[^a-zA-Z0-9]", " ", tweet)

    tweet = re.sub(r" +", " ", tweet)
    # strip trailing and leading whitespaces
    tweet = tweet.strip()
    return tweet


def accuracy_score(preds: torch.Tensor, labels: torch.Tensor):
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return (preds == labels).mean() * 100


def load_data_from_csv(
    filename: str, header=None, cols=None, drop_cols=None, tweets_col=None
) -> pd.DataFrame:
    set_df = pd.read_csv(filename, encoding="latin-1", header=header)
    if cols:
        set_df.columns = cols
    if drop_cols:
        set_df.drop(columns=drop_cols, axis=1, inplace=True)
    if tweets_col:
        set_df[tweets_col] = set_df[tweets_col].apply(clean_tweet)
    return set_df


class BERTSentimentAnalyzer:
    """_summary_

    Args:
        model_name (str, optional): Name of the model, will be used when saving the model. Defaults to "BERT-Sentiment-Analyzer".
        model_path (str, optional): Path to the model that should be loaded, if any. Defaults to None.
        gpu_enabled (bool, optional): Whether or not a gpu is available and should be used. Defaults to True.
    """

    def __init__(
        self,
        model_name: str = "BERT-Sentiment-Analyzer",
        model_path: str = None,
        gpu_enabled: bool = True,
    ):
        self.tokenizer = torch.hub.load(
            "huggingface/pytorch-transformers",
            "tokenizer",
            "bert-base-uncased",
            trust_repo="check",
        )
        self.model = torch.hub.load(
            "huggingface/pytorch-transformers",
            "modelForSequenceClassification",
            "bert-base-uncased",
            output_attentions=True,
            trust_repo="check",
        )

        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 5)

        if gpu_enabled:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model_path = model_path
        self.model_name = model_name

        if self.model_path is not None:
            self.load_model()

        self.model.to(self.device)

    def save_model(self):
        now = dt.now()
        filename = self.model_name + now.strftime("%Y-%m-%d-%H.%M.%S")
        path_cur_file = os.path.dirname(__file__)
        model_path = os.path.join(path_cur_file, "models", filename)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def analyze(self, text: str) -> int:
        text = clean_tweet(text)
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_dict["input_ids"].cuda()
        attention_masks = encoded_dict["attention_mask"].cuda()
        output = self.model(input_ids, attention_masks)
        return output[0].argmax().item()

    def analyze_batch(self, tweets: list):
        sentiments = []
        for tweet in tweets:
            sentiment = self.analyze(tweet)
            sentiments.append(sentiment)
        return sentiments

    def tokenize_data(
        self,
        dataset: pd.DataFrame,
        data_column: str = "OriginalTweet",
        sentiment_column: str = "Sentiment",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_masks = []
        for tweet in dataset[data_column].values:
            encoded_dict = self.tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(dataset[sentiment_column].apply(sentiment_to_int).values)
        return input_ids, attention_masks, labels

    def train(self, train_df: pd.DataFrame, epochs: int = 12):
        # Tokenize all the tweets, mapping words to word IDs.
        input_ids, attention_masks, labels = self.tokenize_data(train_df)

        # Splitting the training data into train and validation sets
        data = TensorDataset(input_ids, attention_masks, labels)
        train_size = int(len(data) * 0.9)
        val_size = len(data) - train_size
        training_data, validation_data = random_split(data, [train_size, val_size])
        # Creating Dataloaders for the training and validation sets
        training_dataloader = DataLoader(
            training_data, sampler=RandomSampler(training_data), batch_size=16
        )
        validation_dataloader = DataLoader(
            validation_data, sampler=RandomSampler(validation_data), batch_size=16
        )

        # Setting up the optimizer and the learning rate scheduler
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(training_dataloader) * epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        torch.cuda.empty_cache()

        # Loss function
        loss_function = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            t_start_epoch = time.time()
            total_loss = 0
            self.model.train()
            for step, batch in enumerate(training_dataloader):
                batch = tuple(t.cuda() for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                )
                loss = loss_function(outputs[0], b_labels)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            total_loss /= len(training_dataloader)

            self.save_model()
            avg_train_loss = total_loss / len(training_dataloader)
            preds, truths, validation_loss = self.evaluate(validation_dataloader)
            accuracy = accuracy_score(preds, truths)
            scheduler.step(validation_loss)
            training_time_epoch = time.time() - t_start_epoch
            print(
                f"Epoch: {epoch + 1},",
                f"Accuracy: {accuracy:.2f}%,",
                f"Average training loss: {avg_train_loss},",
                f"Training time: {training_time_epoch} seconds",
                sep="\t",
            )

    def test(self, test_df: pd.DataFrame):
        input_ids, attention_masks, labels = self.tokenize_data(test_df)
        data = TensorDataset(input_ids, attention_masks, labels)
        test_dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=16
        )
        preds, truths, _ = self.evaluate(test_dataloader)
        accuracy = accuracy_score(preds, truths)
        print(f"Accuracy: {accuracy:.2f}%")

    def evaluate(self, eval_loader: DataLoader):
        self.model.eval()
        predictions, true_labels = [], []
        loss_function = torch.nn.CrossEntropyLoss()
        total_validation_loss = 0
        for batch in eval_loader:
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                )
            logits = outputs[0]
            loss = loss_function(logits, b_labels)
            total_validation_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            predictions.append(logits)
            true_labels.append(label_ids)
        validation_loss = total_validation_loss / len(eval_loader)
        return predictions, true_labels, validation_loss


if __name__ == "__main__":
    TRAIN = False
    TEST = True

    columns = [
        "UserName",
        "ScreenName",
        "Location",
        "TweetAt",
        "OriginalTweet",
        "Sentiment",
    ]
    drop_columns = [
        "UserName",
        "ScreenName",
        "Location",
        "TweetAt",
    ]

    if TEST:
        path_cur_file = os.path.dirname(__file__)
        model_path = os.path.join(path_cur_file, "models/BertModel-acc87.pt")
        model = BERTSentimentAnalyzer(model_path=model_path)
        res = model.analyze("Omg I love this movie so much!")
        print(res)
        res = model.analyze("That ref was really biased")
        print(res)
        test_df = load_data_from_csv(
            "src/data/coronanlp/Corona_NLP_test.csv",
            header=0,
            cols=columns,
            drop_cols=drop_columns,
            tweets_col="OriginalTweet",
        )

        model.test(test_df)

    if TRAIN:
        path_cur_file = os.path.dirname(__file__)
        model_path = os.path.join(path_cur_file, "BertModel2022-12-14-13.40.26")
        model = BERTSentimentAnalyzer(model_name="BertModel", model_path=model_path)
        train_df = load_data_from_csv(
            "src/data/coronanlp/Corona_NLP_train.csv",
            header=0,
            cols=columns,
            drop_cols=drop_columns,
            tweets_col="OriginalTweet",
        )
        model.train(train_df, epochs=100)
