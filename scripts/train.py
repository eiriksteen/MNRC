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

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"USING DEVICE {DEVICE}")


# def mrr_score_old(y_true, y_score):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order)
#     rr_score = y_true / (np.arange(len(y_true)) + 1)
#     return np.sum(rr_score) / np.sum(y_true)

# def mrr_score(y_true, y_score):
#     order = np.argsort(y_score)[::-1]
#     y_true_sorted = np.take(y_true, order)
#     relevant_ranks = np.where(y_true_sorted == 1)[0] + 1
#     if len(relevant_ranks) == 0:
#         return 0.0
#     return np.mean(1 / relevant_ranks)

# def dcg_score(y_true, y_score, k=10):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#     gains = 2 ** y_true - 1
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gains / discounts)

# def ndcg_score(y_true, y_score, k=10):
#     best = dcg_score(y_true, y_true, k)
#     actual = dcg_score(y_true, y_score, k)
#     return actual / best



def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    print("y true shape", np.shape(y_true))
    print("y score shape", np.shape(y_score))

    k = min(len(y_true), k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)



def cal_metric(labels, preds, metrics):
    """Calculate metrics.

    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.
        metrics (list): List of metric names.

    Return:
        dict: Metrics.

    Examples:
        >>> cal_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])
        {'ndcg@2': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    """
    res = {}
    for metric in metrics:
        print("Current metric", metric)
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
                # [mrr_score(labels, preds)]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                    # [ndcg_score(labels, preds, k)]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("Metric {0} not defined".format(metric))
        
        print(res)
    return res

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
                print("batch length")
                print(len(batch))

                user_ids = batch["user_id"].to(DEVICE)
                article_ids = batch["article_id"].to(DEVICE)
                encoded_text = batch["encoded_text"].to(DEVICE) if args.add_text else None
                scores = batch["score"].to(DEVICE)

                logits = model(user_ids, article_ids, encoded_text)
                loss = loss_func(logits, scores)
                valid_loss += loss.item()


                print("logits shape", logits.shape)

                total_preds.append(torch.where(logits > 0.5, 1, 0).squeeze().detach().cpu().tolist())
                total_scores.append(scores.squeeze().detach().cpu().tolist())

        preds_flat = [item for sublist in total_preds for item in sublist]
        scores_flat = [item for sublist in total_scores for item in sublist]
        
        total_preds = total_preds[:-1]
        total_scores = total_scores[:-1]

        # a = accuracy_score(total_scores, total_preds)
        # p, r, f, s = precision_recall_fscore_support(total_scores, total_preds)
        auc = roc_auc_score(scores_flat, preds_flat)
        # mrr = mrr_score(total_scores, total_preds)
        # ndcg = ndcg_score(total_scores, total_preds)
        # dcg = dcg_score(total_scores, total_preds)

        print(total_preds)
        print(len(total_preds))
        print(len(total_preds[0]))

        metrics_dict = cal_metric(total_scores, total_preds, ["ndcg", "mean_mrr"])

        print("Metrics")
        print(metrics_dict)
        
        # metrics = {
        #     "train_loss": train_loss / len(train_loader),
        #     "valid_loss": valid_loss / len(val_loader),
        #     "accuracy": a,
        #     "precision": p.tolist(),
        #     "recall": r.tolist(),
        #     "f1": f.tolist(),
        #     "auc": auc,
        #     "support": s.tolist(),
        #     "mrr": mrr,
        #     "ndcg": ndcg,
        #     "dcg": dcg
        # }

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "valid_loss": valid_loss / len(val_loader),
            "ndcg": metrics_dict["ndcg@1"],
            "auc": auc,
            "auc2": cal_metric(total_scores, total_preds, ["auc"])["auc"]
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

    dataset = MINDDataset(
        "train", 
        to_torch=True, 
        data_path=Path("data")
    )

    num_users = dataset.get_num_users()
    num_items = dataset.get_num_items()
    encoded_texts = dataset.get_article_encoded_texts()

    if args.model == "mf":
        model = MatrixFactorizer(num_users, num_items, args.latent_dim)
    elif args.model == "ncf":
        model = NeuralMatrixFactorizer(num_users, num_items, args.latent_dim)
    else:
        raise ValueError(f"{args.model} not implemented")

    if args.add_text:
        p = f"training_results_{args.model}_{'_wte'}"
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