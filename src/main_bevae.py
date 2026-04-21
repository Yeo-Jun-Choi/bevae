import os
import time
import torch
import argparse
from data import load_data_bevae
from model.BEVAE import BEVAE
from model.trainer import Trainer
from utils import print_args
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='BEVAE Settings')
    parser.add_argument('--dataset', type=str, default='taobao', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_bevae', help='Directory of model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024], help='Hidden layer dimensions (e.g. --hidden_dims 1024 512)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Input dropout rate')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'gelu', 'sigmoid', 'elu', 'leaky_relu'],
                        help='Activation function for encoder/decoder hidden layers')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (number of evaluations without improvement)')
    parser.add_argument('--beta', type=float, default=1e-4, help='KL divergence weight')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 20, 50, 100], help='Top-k items list')
    parser.add_argument('--behavior_weight', type=float, default=1.0,
                        help='Global scalar multiplier applied to all auxiliary behavior weights '
                             '(buy entries are unaffected). Default 1.0 reproduces original behavior.')
    parser.add_argument('--baserate_scale', type=float, default=1.0,
                        help='Scalar multiplier for base rate values. Default 1.0.')
    return parser.parse_args()


def train_epoch(model, data, optimizer, device, beta):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    train_loader = data['train_loader']

    for user_ids in train_loader:
        user_ids = user_ids.to(device)
        loss, recon, kl = model.loss(user_ids, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(train_loader)
    return total_loss / n, total_recon / n, total_kl / n


def main(args):
    print_args(args)

    data = load_data_bevae(
        args.data_dir, args.dataset, args.device, args.batch_size,
        behavior_weight=args.behavior_weight,
        baserate_scale=args.baserate_scale,
    )

    weighted_matrix = data['weighted_matrix']

    model = BEVAE(
        n_items=data['n_items'],
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        input_matrix=weighted_matrix,
        dropout=args.dropout,
        activation=args.activation,
        normalize_input=not args.no_normalize,
        base_rate=data['baserate_vector'],
    ).to(args.device)

    trainer = Trainer(model, data, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)

    if args.load_checkpoint:
        ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
        logger.info(f"Load checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    else:
        logger.info("Start training BEVAE")
        best_ndcg = 0.0
        patience_counter = 0
        train_start = time.time()

        for epoch in range(args.num_epochs):
            epoch_start = time.time()
            loss, recon, kl = train_epoch(model, data, optimizer, args.device, args.beta)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs} ({epoch_time:.1f}s) | "
                f"Loss: {loss:.4f} (Recon: {recon:.4f}, β*KL: {args.beta * kl:.4f})"
            )

            val_results = trainer.evaluate(split='val')
            val_str = ' | '.join(
                f"HR@{k}: {val_results[k]['hr']:.4f}, NDCG@{k}: {val_results[k]['ndcg']:.4f}"
                for k in args.topk
            )
            logger.info(f"  Val:  {val_str}")

            ndcg_10 = val_results[10]['ndcg']
            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                patience_counter = 0
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
                logger.info(f"Best model saved (Val NDCG@10: {best_ndcg:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - train_start
        logger.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}min)")

        ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
        if not os.path.exists(ckpt_path):
            logger.warning("Val NDCG@10 never improved. Saving current model as fallback.")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

    logger.info("Final evaluation")
    results = trainer.evaluate(split='test')
    for k in args.topk:
        logger.info(f"Test HR@{k}: {results[k]['hr']:.4f}, NDCG@{k}: {results[k]['ndcg']:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
