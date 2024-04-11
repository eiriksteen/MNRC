from torch.utils.data import DataLoader
from pathlib import Path
from mnrc.torch_datasets import MINDDataset
from mnrc.models import MatrixFactorizer

if __name__ == "__main__":

    dataset = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data")
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    num_users = dataset.get_num_users()
    num_items = dataset.get_num_items()
    model = MatrixFactorizer(num_users, num_items, 64)
    preds = model(batch["user_id"], batch["article_id"])

    print(batch)
    print(preds)