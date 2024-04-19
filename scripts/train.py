import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from pathlib import Path
from mnrc.torch_datasets import MINDDataset
from mnrc.models import MatrixFactorizer, NeuralMatrixFactorizer
from pprint import pprint
import numpy as np

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"USING DEVICE {DEVICE}")


def mrr_score_old(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

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

def train(
        model, 
        train_data, 
        validation_data,
        args,
        out_dir
    ):

    print("TRAINING MODEL")

    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, args.batch_size)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    min_val_loss = float("inf")

    for epoch in range(args.num_epochs): 
        print(f"RUNNING EPOCH {epoch+1}")
        
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            
            user_ids = batch["user_id"].to(DEVICE)
            article_ids = batch["article_id"].to(DEVICE)
            encoded_text = batch["encoded_text"].to(DEVICE) if args.add_text else None
            scores = batch["score"].to(DEVICE)

            logits = model(user_ids, article_ids, encoded_text)
            loss = loss_func(logits, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        total_preds, total_scores = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader):

                user_ids = batch["user_id"].to(DEVICE)
                article_ids = batch["article_id"].to(DEVICE)
                encoded_text = batch["encoded_text"].to(DEVICE) if args.add_text else None
                scores = batch["score"].to(DEVICE)

                logits = model(user_ids, article_ids, encoded_text)
                loss = loss_func(logits, scores)
                valid_loss += loss.item()

                total_preds += torch.where(logits > 0.5, 1, 0).squeeze().detach().cpu().tolist()
                total_scores += scores.squeeze().detach().cpu().tolist()

        a = accuracy_score(total_scores, total_preds)
        p, r, f, s = precision_recall_fscore_support(total_scores, total_preds)
        auc = roc_auc_score(total_scores, total_preds)
        mrr = mrr_score(total_scores, total_preds)
        ndcg = ndcg_score(total_scores, total_preds)
        dcg = dcg_score(total_scores, total_preds)
        
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "valid_loss": valid_loss / len(val_loader),
            "accuracy": a,
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "auc": auc,
            "support": s.tolist(),
            "mrr": mrr,
            "ndcg": ndcg,
            "dcg": dcg
        }

        if metrics["valid_loss"] < min_val_loss:
            print("New min loss, saving model")
            min_val_loss = metrics["valid_loss"]
            torch.save(model.state_dict(), out_dir / "model")

            with open(out_dir / f"metrics_epoch{epoch+1}.json", "w") as f:
                json.dump(metrics, f)

        pprint(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="mf", type=str)
    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("--lr", default=1e-03, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--from_scratch", action=argparse.BooleanOptionalAction)
    parser.add_argument("--add_text", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    train_data = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data")
    )

    validation_data = MINDDataset(
        "validation", 
        to_torch=True, 
        data_path=Path("data")
    )

    num_users = train_data.get_num_users()
    num_items = train_data.get_num_items()
    encoded_texts = train_data.get_article_encoded_texts()

    if args.model == "mf":
        model = MatrixFactorizer(num_users, num_items, args.latent_dim)
    elif args.model == "ncf":
        model = NeuralMatrixFactorizer(
            num_users, 
            num_items, 
            args.latent_dim,
            args.add_text
        )
    else:
        raise ValueError(f"{args.model} not implemented")

    if args.add_text:
        p = f"training_results_{args.model}_wte_new"
    else:
        p = f"training_results_{args.model}"

    out_dir = Path(p)
    out_dir.mkdir(exist_ok=True)

    if not args.from_scratch:
        try:
            model.load_state_dict(torch.load(out_dir / "model"))
            print(f"Model found at {out_dir / 'model'}, resuming training")
        except FileNotFoundError:
            print(f"No model found at {out_dir / 'model'}, training from scratch")

    train(
        model,
        train_data,
        validation_data,
        args,
        out_dir=out_dir
    )