import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class MINDDataset(Dataset):

    def __init__(
            self,
            split: str,
            to_torch: bool = True,
            data_path: Path = Path("data")
        ):

        if split == "train":
            self.behaviors_df = pd.read_csv(data_path / "MINDsmall_train" / "behaviors.tsv", sep="\t")
            self.news_df = pd.read_csv(data_path / "MINDsmall_train" / "news.tsv", sep="\t")
        elif split == "validation":
            self.behaviors_df = pd.read_csv(data_path / "MINDsmall_dev" / "behaviors.tsv", sep="\t")
            self.news_df = pd.read_csv(data_path / "MINDsmall_dev" / "news.tsv", sep="\t")
        else:
            raise ValueError("Split must be train or validation")
        
        self.behaviors_df.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
        self.news_df.columns = ["News ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title Entities", "Abstract Entities"]

        self.to_torch = to_torch
        self.user_to_id, self.id_to_user, self.article_to_id, self.id_to_article = self.get_id_dicts()

        self.behaviors = self.preprocess_behaviors()

    def __len__(self):

        return len(self.behaviors)
    
    def __getitem__(self, index):

        key, score = self.behaviors[index]
        user, article, _ = key.split("-")
        user_id, article_id = self.user_to_id[user], self.article_to_id[article]

        if self.to_torch:
            user_id = torch.Tensor([user_id]).long()
            article_id = torch.Tensor([article_id]).long()
            score = torch.Tensor([score]).float()

        sample = {
            "user_id": user_id,
            "article_id": article_id,
            "score": score
        }   
        
        return sample
    
    def get_num_users(self):
        return len(self.user_to_id)
    
    def get_num_items(self):
        return len(self.article_to_id)

    def get_id_dicts(self):
        
        users = self.behaviors_df["User ID"].unique().tolist()
        articles = self.news_df["News ID"].unique().tolist()

        user_to_id = {user: i for i, user in enumerate(users)}
        id_to_user = {i: user for i, user in enumerate(users)}
        article_to_id = {article: i for i, article in enumerate(articles)}
        id_to_article = {i: article for i, article in enumerate(articles)}

        return user_to_id, id_to_user, article_to_id, id_to_article
    
    def preprocess_behaviors(self):

        behaviors_dict = {}

        for _, row in self.behaviors_df.iterrows():
            user = row["User ID"]
            for impression in row["Impressions"].split():
                article, label = impression.split("-")
                behaviors_dict[f"{user}-{article}-{row['Time']}"] = float(label)
            if not pd.isna(row["History"]):
                for article in row["History"].split():
                    behaviors_dict[f"{user}-{article}-{row['Time']}"] = 1.0

        return list(behaviors_dict.items())
