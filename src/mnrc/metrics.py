from mnrc.models import MatrixFactorizer, NeuralMatrixFactorizer
import pandas as pd
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from mnrc.torch_datasets import MINDDataset


def process_impression(s: str):
    list_of_strings = s.split(" ")
    itemid_rel_tuple = [l.split("-") for l in list_of_strings]
    click = None
    noclicks = []
    for entry in itemid_rel_tuple:
        if entry[1] =='0':
            noclicks.append(entry[0])
        if entry[1] =='1':
            click = entry[0]
    return noclicks, click

def get_impression_df(self, data_path: Path = Path("data"), split: str = "MINDsmall_validation"):
    behaviors_df = pd.read_csv(data_path / split / "behaviors.tsv", sep="\t")
    behaviors_df.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
    
    behaviors_df['Noclicks'], behaviors_df['Click'] = zip(*behaviors_df['Impressions'].map(process_impression))

    return behaviors_df

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    relevant_ranks = np.where(y_true_sorted == 1)[0] + 1
    if len(relevant_ranks) == 0:
        return 0.0
    return np.mean(1 / relevant_ranks)

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def get_article_texts(article_ids, data_path: Path = Path("data")):
    news_df = pd.read_csv(data_path / "MINDsmall_dev" / "news.tsv", sep="\t")
    news_df.columns = ["News ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title Entities", "Abstract Entities"]
    news_df["All Text"] = news_df["Title"]+ \
                        ". "+news_df["Category"]+ \
                            ". "+news_df["SubCategory"]+ \
                                ". "+news_df["Abstract"]
    news_df = news_df.set_index("News ID")
    texts = [news_df.loc[article_id]["All Text"] for article_id in article_ids]
    texts = [t if not pd.isna(t) else "" for t in texts]
    return texts

def cal_metrics(model, text_encoder, user_to_id, article_to_id, data_path: Path = Path("data"), to_torch: bool = True, split: str = "MINDsmall_validation"):
    behaviors_df_val = get_impression_df(data_path, split="MINDsmall_validation")

    for _, row in behaviors_df_val.iterrows():
        click = row.Click
        noclicks = row.Noclicks
        user_id = row['User ID']
        y_true = [1] + [0]*len(noclicks)
        article_ids = [click] + [0]*len(noclicks)
        article_ids = [article_to_id[article] for article in article_ids]
        encoded_texts = text_encoder.encode(
            get_article_texts(article_ids=article_ids, data_path=data_path),
            convert_to_tensor=True,
            show_progress_bar=True).detach().cpu().numpy()
        user_id = user_to_id[user_id]

        if to_torch:
            user_id = torch.Tensor([user_id]).long()
            article_ids = torch.Tensor(article_ids).long()
            y_true = torch.Tensor(y_true).float()

        y_score = []
        for article_id, encoded_text in zip(article_ids, encoded_texts):
            y_score.append(model(user_id, article_id, encoded_text).item())
        
    return click, noclicks

if __name__ == "__main__":
    behaviors_df = get_impression_df("validation", Path("data"))
    print(behaviors_df[["Noclicks", "Click"]])