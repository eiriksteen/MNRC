import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from mnrc.torch_datasets import MINDDataset
from mnrc.models import MatrixFactorizer
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
        lr,
        batch_size,
        num_epochs
    ):

    model = model.to(DEVICE)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"RUNNING EPOCH {epoch+1}")
        
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            
            user_ids = batch["user_id"].to(DEVICE)
            article_ids = batch["article_id"].to(DEVICE)
            scores = batch["score"].to(DEVICE)

            logits = model(user_ids, article_ids)
            loss = loss_func(logits, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):

                user_ids = batch["user_id"].to(DEVICE)
                article_ids = batch["article_id"].to(DEVICE)
                scores = batch["score"].to(DEVICE)

                logits = model(user_ids, article_ids)
                loss = loss_func(logits, scores)
                valid_loss += loss.item()

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "valid_loss": valid_loss / len(val_loader)
        }

        pprint(metrics)


if __name__ == "__main__":

    latent_dim = 64
    out_dir = Path("training_results")

    dataset = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data")
    )

    num_users = dataset.get_num_users()
    num_items = dataset.get_num_items()

    model = MatrixFactorizer(num_users, num_items, latent_dim)

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
        lr=1e-03,
        batch_size=64,
        num_epochs=25
    )