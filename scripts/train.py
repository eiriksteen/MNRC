import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from mnrc.torch_datasets import MINDDataset
from mnrc.models import MatrixFactorizer, NeuralMatrixFactorizer
from mnrc.metrics import compute_metrics
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

    print("TRAINING MODEL")

    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    loss_func = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([24.0 if args.weighted else 1.0]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    max_auc = 0

    for epoch in range(args.num_epochs): 
        print(f"RUNNING EPOCH {epoch+1}")
        
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            
            user_ids = batch["user_id"].to(DEVICE)
            article_ids = batch["article_id"].to(DEVICE)
            encoded_text = batch["encoded_text"].to(DEVICE) if args.add_text else None
            scores = batch["score"].to(DEVICE)

            logits, _ = model(user_ids, article_ids, encoded_text)
            loss = loss_func(logits, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        metrics = compute_metrics(
            model, 
            validation_data, 
            loss_func, 
            DEVICE)
        metrics["train_loss"] = train_loss / len(train_loader)

        if metrics["auc"] > max_auc:
            print("New max auc, saving model")
            max_auc = metrics["auc"]
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
    parser.add_argument("--weighted", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    train_data = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data"),
        include_encoded_text=args.add_text
    )

    validation_data = MINDDataset(
        "validation", 
        to_torch=True, 
        data_path=Path("data"),
        include_encoded_text=args.add_text
    )

    num_users = train_data.get_num_users()
    num_items = train_data.get_num_items()

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
        p = f"training_results_{args.model}_wte"
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