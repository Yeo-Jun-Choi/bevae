import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class BPRDataset(Dataset):
    def __init__(self, buy_interactions, n_items):
        self.buy_interactions = {int(k): set(v) for k, v in buy_interactions.items()}
        self.all_items = set(range(1, n_items + 1))
        self.users = list(self.buy_interactions.keys())
        self.total_samples = []
        for user in self.users:
            for pos_item in self.buy_interactions[user]:
                self.total_samples.append((user, pos_item))

    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        neg_candidates = list(self.all_items - self.buy_interactions[user])
        neg_item = random.choice(neg_candidates)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


class BEVAEDataset(Dataset):
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.user_ids[idx], dtype=torch.long)


def _detect_behaviors(data_dir):
    """Auto-detect available behavior types from files in the data directory."""
    candidates = ['buy', 'view', 'click', 'cart', 'collect']
    return [b for b in candidates if os.path.exists(f'{data_dir}/{b}.txt')]


def _compute_behavior_weights(data_dir, aux_behaviors, train_buy):
    """Target-intersected ratio per auxiliary behavior: |E_b ∩ E_buy| / |E_b|."""
    buy_lookup = {int(u): set(items) for u, items in train_buy.items()}
    weights = {}
    for b in aux_behaviors:
        edge_raw = np.loadtxt(f'{data_dir}/{b}.txt', dtype=int)
        n_b = len(edge_raw)
        if n_b == 0:
            weights[b] = 0.0
            continue
        users, items = edge_raw[:, 0], edge_raw[:, 1]
        mask = np.array([item in buy_lookup.get(u, set()) for u, item in zip(users, items)])
        weights[b] = mask.sum() / n_b
    return weights


def _build_weighted_matrix(data_dir, n_users, n_items, aux_behaviors, train_buy, behavior_weight):
    """Build the weighted user-item matrix used as the VAE input/target.

    Auxiliary behaviors are weighted by target-intersected ratio (scaled by
    `behavior_weight`); buy entries are overwritten to 1.0.
    """
    mat_size = (n_users + 1, n_items + 1)
    behavior_weights = _compute_behavior_weights(data_dir, aux_behaviors, train_buy)
    if behavior_weight != 1.0:
        behavior_weights = {b: w * behavior_weight for b, w in behavior_weights.items()}

    rows_list, cols_list, vals_list = [], [], []
    for b in aux_behaviors:
        w = behavior_weights[b]
        if w == 0.0:
            continue
        edge_raw = np.loadtxt(f'{data_dir}/{b}.txt', dtype=int)
        rows_list.append(edge_raw[:, 0])
        cols_list.append(edge_raw[:, 1])
        vals_list.append(np.full(len(edge_raw), w))

    if rows_list:
        rows = torch.from_numpy(np.concatenate(rows_list)).long()
        cols = torch.from_numpy(np.concatenate(cols_list)).long()
        vals = torch.from_numpy(np.concatenate(vals_list)).float()
        dense = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=mat_size
        ).coalesce().to_dense()
    else:
        dense = torch.zeros(mat_size)

    buy_rows, buy_cols = [], []
    for u, items in train_buy.items():
        uu = int(u)
        for i in items:
            buy_rows.append(uu)
            buy_cols.append(i)
    buy_rows_t = torch.tensor(buy_rows, dtype=torch.long)
    buy_cols_t = torch.tensor(buy_cols, dtype=torch.long)
    dense[buy_rows_t, buy_cols_t] = 1.0

    weighted = dense.to_sparse_coo().coalesce()
    return weighted, behavior_weights


def _compute_baserate_vector(data_dir, aux_behaviors, train_buy, n_users, n_items):
    """Per-user base-rate vector.

    base_rate(u) = |unvisit_buy(u)| / |non_interacted(u)|, where:
        unvisit_buy(u)     = items bought by u without any auxiliary behavior
        non_interacted(u)  = items with no interaction at all
    """
    aux_lookup = {}
    for b in aux_behaviors:
        edge_raw = np.loadtxt(f'{data_dir}/{b}.txt', dtype=int)
        for u, i in edge_raw:
            aux_lookup.setdefault(int(u), set()).add(int(i))

    min_rate = 1.0 / n_items
    base_rate = torch.full((n_users + 1,), min_rate)
    base_rate[0] = 0.0
    for u_str, buy_items in train_buy.items():
        u = int(u_str)
        buy_set = set(buy_items)
        aux_set = aux_lookup.get(u, set())
        all_interacted = buy_set | aux_set
        unvisit_buy = buy_set - aux_set
        non_interacted = n_items - len(all_interacted)
        if non_interacted > 0 and len(unvisit_buy) > 0:
            base_rate[u] = max(len(unvisit_buy) / non_interacted, min_rate)
    return base_rate


def load_data_bevae(data_dir, dataset, device, batch_size,
                    behavior_weight=1.0, baserate_scale=1.0):
    logger.info('Load data (BEVAE)')
    data_dir = os.path.join(data_dir, dataset)

    with open(f'{data_dir}/count.txt', 'r') as f:
        count = json.load(f)
    n_users, n_items = count['user'], count['item']

    behaviors = _detect_behaviors(data_dir)
    aux_behaviors = [b for b in behaviors if b != 'buy']
    logger.info(f"Detected behaviors: {behaviors} | Users: {n_users}, Items: {n_items}")

    with open(f'{data_dir}/buy_dict.txt', 'r') as f:
        train_buy = json.load(f)
    with open(f'{data_dir}/validation_dict.txt', 'r') as f:
        val_buy = json.load(f)
    with open(f'{data_dir}/test_dict.txt', 'r') as f:
        test_buy = json.load(f)

    weighted_matrix, behavior_weights = _build_weighted_matrix(
        data_dir, n_users, n_items, aux_behaviors, train_buy, behavior_weight,
    )
    weighted_matrix = weighted_matrix.to(device)
    logger.info(f"behavior_weight: {behavior_weight} | per-behavior weights: {behavior_weights}")

    baserate_vector = _compute_baserate_vector(
        data_dir, aux_behaviors, train_buy, n_users, n_items,
    )
    baserate_vector *= baserate_scale
    baserate_vector = baserate_vector.to(device)
    nonzero = (baserate_vector > 0).sum().item()
    logger.info(
        f"Base rate (scale={baserate_scale}): {nonzero}/{n_users} users with non-zero, "
        f"mean={baserate_vector[baserate_vector > 0].mean().item():.6f}, "
        f"max={baserate_vector.max().item():.6f}"
    )

    train_user_ids = [int(u) for u in train_buy.keys()]
    train_dataset = BEVAEDataset(train_user_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=4)

    val_dataset = BPRDataset(val_buy, n_items)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
    val_gt_length = np.array([len(items) for items in val_buy.values()])

    test_dataset = BPRDataset(test_buy, n_items)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=4)
    test_gt_length = np.array([len(items) for items in test_buy.values()])

    return {
        'n_users': n_users,
        'n_items': n_items,
        'weighted_matrix': weighted_matrix,
        'baserate_vector': baserate_vector,
        'behavior_weights': behavior_weights,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'val_gt': val_buy,
        'val_gt_length': val_gt_length,
        'test_loader': test_loader,
        'train_gt': train_buy,
        'test_gt': test_buy,
        'test_gt_length': test_gt_length,
    }
