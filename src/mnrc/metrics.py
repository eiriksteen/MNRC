import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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

def compute_metrics(model, dataset, loss_func, device="cpu"):

    model.eval()
    impressions = dataset.get_impressions()
    group_scores, group_labels = [], []
    loss = 0

    with torch.no_grad():
        
        for user, imp in tqdm(impressions):
            # Parse row
            articles = [imp_str.split("-")[0] for imp_str in imp.split()]
            labels = [float(imp_str.split("-")[1]) for imp_str in imp.split()]
            user_id = dataset.user_to_id[user]
            article_ids = [dataset.article_to_id[article] for article in articles]

            # Tensorify
            user_id_tensor = torch.Tensor([user_id for _ in range(len(labels))]).long().to(device)
            article_id_tensor = torch.Tensor(article_ids).long().to(device)
            labels_tensor = torch.Tensor(labels).to(device)

            if model.text_projector is not None:
                encoded_text = torch.Tensor(dataset.encoded_texts[article_ids]).to(device)
            else:
                encoded_text = None

            # Get preds
            logits, scores = model(user_id_tensor, article_id_tensor, encoded_text)
            logits, scores = logits.squeeze(dim=-1), scores.squeeze(dim=-1)
            loss += loss_func(logits, labels_tensor)
            group_scores.append(scores.detach().cpu().tolist())
            group_labels.append(labels_tensor.detach().cpu().tolist())
        
    
    ndcg5 = np.mean(
        [ndcg_score(labels, scores, 5) for labels, scores in zip(group_labels, group_scores)]
    )

    ndcg10 = np.mean(
        [ndcg_score(labels, scores, 10) for labels, scores in zip(group_labels, group_scores)]
    )

    mrr = np.mean(
        [mrr_score(labels, scores) for labels, scores in zip(group_labels, group_scores)]
    )

    group_auc = np.mean(
        [roc_auc_score(labels, scores) for labels, scores in zip(group_labels, group_scores)]
    )

    group_scores_flat = np.asarray([s for g in group_scores for s in g])
    group_preds = np.where(group_scores_flat > 0.5, 1.0, 0.0)
    labels_flat = np.asarray([l for g in group_labels for l in g])
    a = accuracy_score(labels_flat, group_preds)
    p, r, f, s = precision_recall_fscore_support(labels_flat, group_preds)

    return {
        "ndcg5": ndcg5,
        "ndcg10": ndcg10,
        "mrr": mrr,
        "auc": group_auc,
        "accuracy": a,
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f.tolist(),
        "support": s.tolist(),
        "val_loss": loss.item() / len(group_labels)
    }
