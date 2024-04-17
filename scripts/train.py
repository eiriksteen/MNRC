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

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"USING DEVICE {DEVICE}")

def train(
        model, 
        train_data, 
        validation_data,
        args,
        out_dir
    ):

    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, args.batch_size)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    min_val_loss = float("inf")
    add_text = args.add_embeds_from_text_pca or args.add_embeds_from_text_linear

    for epoch in range(args.num_epochs): 
        print(f"RUNNING EPOCH {epoch+1}")
        
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            
            user_ids = batch["user_id"].to(DEVICE)
            article_ids = batch["article_id"].to(DEVICE)
            encoded_text = batch["encoded_text"].to(DEVICE) if add_text else None
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
                encoded_text = batch["encoded_text"].to(DEVICE) if add_text else None
                scores = batch["score"].to(DEVICE)

                logits = model(user_ids, article_ids, encoded_text)
                loss = loss_func(logits, scores)
                valid_loss += loss.item()

                total_preds += torch.where(logits > 0.5, 1, 0).squeeze().detach().cpu().tolist()
                total_scores += scores.squeeze().detach().cpu().tolist()

        a = accuracy_score(total_scores, total_preds)
        p, r, f, s = precision_recall_fscore_support(total_scores, total_preds)
        auc = roc_auc_score(total_scores, total_preds)
        
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "valid_loss": valid_loss / len(val_loader),
            "accuracy": a,
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "auc": auc,
            "support": s.tolist()
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
    parser.add_argument("--add_embeds_from_text_pca", action=argparse.BooleanOptionalAction)
    parser.add_argument("--add_embeds_from_text_linear", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    dataset = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data")
    )

    num_users = dataset.get_num_users()
    num_items = dataset.get_num_items()
    encoded_texts = dataset.get_article_encoded_texts()

    if args.model == "mf":
        model = MatrixFactorizer(num_users, num_items, args.latent_dim, args.add_embeds_from_text_linear)
    elif args.model == "ncf":
        model = NeuralMatrixFactorizer(num_users, num_items, args.latent_dim, args.add_embeds_from_text_linear)
    else:
        raise ValueError(f"{args.model} not implemented")

    p = f"training_results_{args.model}"
    if args.add_embeds_from_text_pca:
        p = f"training_results_{args.model}_{'_pca'}"
    elif args.add_embeds_from_text_linear:
        p = f"training_results_{args.model}_{'_linear'}"
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

    train_data, validation_data = random_split(dataset, [0.8, 0.2])

    train(
        model,
        train_data,
        validation_data,
        args,
        out_dir=out_dir
    )