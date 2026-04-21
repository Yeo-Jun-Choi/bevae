import torch
import numpy as np

from .metrics import ndcg, hit


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk if isinstance(args.topk, list) else [args.topk]

    def evaluate(self, split='test'):
        """Evaluate on the given split ('val' or 'test')."""
        device = self.args.device
        max_k = max(self.topk)

        gt_key = 'val_gt' if split == 'val' else 'test_gt'
        gt_dict = self.data[gt_key]
        train_gt = self.data['train_gt']
        val_gt = self.data.get('val_gt', {})
        test_users = [int(u) for u in gt_dict.keys()]

        self.model.eval()
        topk_masks = []
        gt_lengths_all = []
        num_items = self.model.input_matrix.shape[1] if hasattr(self.model, 'input_matrix') else None
        with torch.no_grad():
            for i in range(0, len(test_users), 512):
                batch = test_users[i:i + 512]
                B = len(batch)
                user_tensor = torch.tensor(batch, dtype=torch.long, device=device)
                scores = self.model.predict(user_tensor)
                if num_items is None:
                    num_items = scores.shape[1]

                # Vectorized masking: build flat (row, col) index lists for items to mask.
                mask_rows = []
                mask_cols = []
                gt_rows = []
                gt_cols = []
                gt_lengths = np.empty(B, dtype=np.int64)
                for idx, user in enumerate(batch):
                    su = str(user)
                    tr = train_gt.get(su)
                    if tr:
                        mask_rows.append(np.full(len(tr), idx, dtype=np.int64))
                        mask_cols.append(np.asarray(tr, dtype=np.int64))
                    if split == 'test':
                        vi = val_gt.get(su)
                        if vi:
                            mask_rows.append(np.full(len(vi), idx, dtype=np.int64))
                            mask_cols.append(np.asarray(vi, dtype=np.int64))
                    gi = gt_dict[su]
                    gt_rows.append(np.full(len(gi), idx, dtype=np.int64))
                    gt_cols.append(np.asarray(gi, dtype=np.int64))
                    gt_lengths[idx] = len(gi)

                if mask_rows:
                    mr = torch.from_numpy(np.concatenate(mask_rows)).to(device)
                    mc = torch.from_numpy(np.concatenate(mask_cols)).to(device)
                    scores.index_put_((mr, mc), torch.full((mr.numel(),), float('-inf'), device=device))

                _, topk_indices = torch.topk(scores, max_k, dim=1)  # (B, max_k)

                gt_mask = torch.zeros((B, num_items), dtype=torch.bool, device=device)
                gr = torch.from_numpy(np.concatenate(gt_rows)).to(device)
                gc = torch.from_numpy(np.concatenate(gt_cols)).to(device)
                gt_mask[gr, gc] = True
                hits = torch.gather(gt_mask, 1, topk_indices)  # (B, max_k) bool

                topk_masks.append(hits.cpu().numpy())
                gt_lengths_all.append(gt_lengths)

        topk_list = np.vstack(topk_masks)
        gt_lengths = np.concatenate(gt_lengths_all)
        hr_res = hit(topk_list, gt_lengths).mean(axis=0)
        ndcg_res = ndcg(topk_list, gt_lengths).mean(axis=0)

        results = {}
        for k in self.topk:
            results[k] = {'hr': hr_res[k-1], 'ndcg': ndcg_res[k-1]}

        return results
