import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class MINDDataset(Dataset):

    def __init__(
            self,
            split: str,
            to_torch: bool = True,
            data_path: Path = Path("data"),
            include_encoded_text: bool = True
        ):

        print(f"Loading dataset from {data_path}")
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
        self.include_encoded_text = include_encoded_text
        self.user_to_id, self.id_to_user, self.article_to_id, self.id_to_article = self.get_id_dicts()
        self.behaviors_processed = self.preprocess_behaviors()

        self.news_df["All Text"] = self.news_df["Title"]+ \
                                ". "+self.news_df["Category"]+ \
                                    ". "+self.news_df["SubCategory"]+ \
                                        ". "+self.news_df["Abstract"]
        
        if include_encoded_text:
            self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.encoded_texts = self.text_encoder.encode(
                self.get_article_texts(),
                convert_to_tensor=True,
                show_progress_bar=True
            ).detach().cpu().numpy()

    def __len__(self):
        return len(self.behaviors_processed)
    
    def __getitem__(self, index):

        key, score = self.behaviors_processed[index]
        user, article, _ = key.split("-")
        user_id, article_id = self.user_to_id[user], self.article_to_id[article]
        encoded_text = self.encoded_texts[article_id] if self.include_encoded_text else None

        if self.to_torch:
            user_id = torch.Tensor([user_id]).long()
            article_id = torch.Tensor([article_id]).long()
            score = torch.Tensor([score]).float()
            encoded_text = torch.Tensor(encoded_text) if self.include_encoded_text else None

        sample = {
            "user_id": user_id,
            "article_id": article_id,
            "score": score
        }   

        if self.include_encoded_text:
            sample["encoded_text"] = encoded_text
        
        return sample
    
    def get_num_users(self):
        return len(self.user_to_id)
    
    def get_num_items(self):
        return len(self.article_to_id)
    
    def get_article_texts(self):
        texts = [t if not pd.isna(t) else "" for t in self.news_df["All Text"].tolist()]
        assert len(texts) == self.get_num_items()
        return texts
    
    def get_article_encoded_texts(self):
        return self.encoded_texts

    def get_id_dicts(self):
        
        users = self.behaviors_df["User ID"].unique().tolist()
        articles = self.news_df["News ID"].unique().tolist()

        user_to_id = {user: i for i, user in enumerate(users)}
        id_to_user = {i: user for i, user in enumerate(users)}
        article_to_id = {article: i for i, article in enumerate(articles)}
        id_to_article = {i: article for i, article in enumerate(articles)}

        return user_to_id, id_to_user, article_to_id, id_to_article
    
    def get_impressions(self):
        return self.behaviors_df[["User ID", "Impressions"]].values.tolist()
    
    def preprocess_behaviors(self):

        behaviors_dict = {}

        counts = [0,0]
        for _, row in self.behaviors_df.iterrows():
            user = row["User ID"]
            for impression in row["Impressions"].split():
                article, label = impression.split("-")
                if article in self.article_to_id.keys():
                    behaviors_dict[f"{user}-{article}-{row['Time']}"] = float(label)
                    counts[int(label)]+=1

            # if not pd.isna(row["History"]):
            #     for article in row["History"].split():
            #         if article in self.article_to_id.keys():
            #             behaviors_dict[f"{user}-{article}-{row['Time']}"] = 1.0

        print(f"LABEL COUNTS = {counts}")
        return list(behaviors_dict.items())
