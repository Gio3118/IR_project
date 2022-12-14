import os
import torch
import pandas as pd
import re
import time
import numpy as np
from datetime import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

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
            trust_repo=True,
        )
        self.model = torch.hub.load(
            "huggingface/pytorch-transformers",
            "modelForSequenceClassification",
            "bert-base-uncased",
            output_attentions=True,
            trust_repo=True,
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
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )

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
        input_ids = encoded_dict["input_ids"].to(self.device)
        attention_masks = encoded_dict["attention_mask"].to(self.device)
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
        # optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(training_dataloader) * epochs
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.1, patience=5
        # )
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
            # scheduler.step(validation_loss)
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
        return preds, truths

    def evaluate(self, eval_loader: DataLoader):
        torch.cuda.empty_cache()
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


def tweet_eval():
    path_cur_file = os.path.dirname(__file__)
    model_path = os.path.join(path_cur_file, "models/BertModel-acc87.pt")
    model = BERTSentimentAnalyzer(model_path=model_path)

    test_df = load_data_from_csv(
        "src/data/tweeteval/test_text (1).csv",
        header=0,
        cols=["OriginalTweet", "Sentiment"],
        drop_cols=[],
        tweets_col="OriginalTweet",
    )

    preds, labels = model.test(test_df)

    targets = [
        "Negative",
        "Neutral",
        "Positive",
    ]
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    preds[preds == 1] = 0
    preds[preds == 2] = 1
    preds[preds == 3] = 2
    preds[preds == 4] = 2
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    labels[labels == 3] = 2
    labels[labels == 4] = 2
    report = classification_report(labels, preds, target_names=targets)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    print(report)
    df = pd.DataFrame(cm, index=targets, columns=targets)
    heatmap = sns.heatmap(df, annot=True, fmt="g")
    heatmap.set(xlabel="Predicted", ylabel="Actual")
    heatmap.set_title("Confusion Matrix of TweetEval Test Set")
    plt.show()


def get_metrics(model, dataloader, targets):
    preds, labels, _ = model.evaluate(dataloader)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    report = classification_report(
        labels, preds, target_names=targets, output_dict=True
    )
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    print(report)
    return precision, recall, f1, _


def generate_graphs(cols, dropcols):
    precision_test = []
    recall_test = []
    f1_test = []
    loss_test = []
    precision_train = []
    recall_train = []
    f1_train = []
    loss_train = []
    targets = [
        "E. Negative",
        "Negative",
        "Neutral",
        "Positive",
        "E. Positive",
    ]
    for i in range(1, 13):

        path_cur_file = os.path.dirname(__file__)
        model_path = os.path.join(path_cur_file, f"models/old/model_{i}.pt")
        model = BERTSentimentAnalyzer(model_path=model_path)
        test_df = load_data_from_csv(
            "src/data/coronanlp/Corona_NLP_test.csv",
            header=0,
            cols=cols,
            drop_cols=dropcols,
            tweets_col="OriginalTweet",
        )

        input_ids, attention_masks, labels = model.tokenize_data(test_df)
        data = TensorDataset(input_ids, attention_masks, labels)
        test_dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=16
        )

        train_df = load_data_from_csv(
            "src/data/coronanlp/Corona_NLP_train.csv",
            header=0,
            cols=cols,
            drop_cols=dropcols,
            tweets_col="OriginalTweet",
        )

        input_ids, attention_masks, labels = model.tokenize_data(train_df)
        data = TensorDataset(input_ids, attention_masks, labels)
        train_dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=16
        )

        print("Epoch", i)
        t0 = time.perf_counter()
        # model_path = os.path.join(path_cur_file, f"models/old/model_{i}.pt")
        # model.model_path = model_path
        # model.load_model()

        p, r, f, loss = get_metrics(model, test_dataloader, targets)
        precision_test.append(p)
        recall_test.append(r)
        f1_test.append(f)
        loss_test.append(loss)

        p, r, f, loss = get_metrics(model, train_dataloader, targets)
        precision_train.append(p)
        recall_train.append(r)
        f1_train.append(f)
        loss_train.append(loss)
        print("Epoch: ", i, "time: {:.2f}".format(time.perf_counter() - t0))

    print("Precision Test:", precision_test)
    print("Recall Test:", recall_test)
    print("F1 Test:", f1_test)
    print("Loss Test:", loss_test)
    print("Precision Train:", precision_train)
    print("Recall Train:", recall_train)
    print("F1 Train:", f1_train)
    print("Loss Train:", loss_train)


if __name__ == "__main__":
    TRAIN = False
    TEST = True
    GENERATE_GRAPHS = False

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

    if GENERATE_GRAPHS:
        # generate_graphs(columns, drop_columns)
        # output_graphs()
        pass

    if TEST:
        path_cur_file = os.path.dirname(__file__)
        model_path = os.path.join(path_cur_file, "models/old/BertModel-Epoch-5.pt")
        model = BERTSentimentAnalyzer(model_path=model_path)
        test_df = load_data_from_csv(
            "src/data/coronanlp/Corona_NLP_test.csv",
            header=0,
            cols=columns,
            drop_cols=drop_columns,
            tweets_col="OriginalTweet",
        )

        preds, labels = model.test(test_df)

        targets = [
            "E. Negative",
            "Negative",
            "Neutral",
            "Positive",
            "E. Positive",
        ]
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        report = classification_report(labels, preds, target_names=targets)
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
        df = pd.DataFrame(cm, index=targets, columns=targets)
        heatmap = sns.heatmap(df, annot=True, fmt="g")
        heatmap.set(xlabel="Predicted", ylabel="Actual")
        heatmap.set_title("Confusion Matrix of CoronaNLP Test Set")
        plt.show()

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
