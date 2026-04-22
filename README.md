# BE-VAE

Official implementation of **BE-VAE**, a Variational Autoencoder for multi-behavior recommendation.


---

## Dependencies

- Python 3.9+
- CUDA 11.8-compatible GPU (for the default `requirements.txt`)
- PyTorch 2.0.1 + PyG stack, loguru, numpy, scipy, scikit-learn


```bash
pip install -r requirements.txt
```


## Datasets

Three preprocessed e-commerce datasets are included under `data/`:

| Dataset | # Users | # Items | Behaviors                |
|---------|--------:|--------:|--------------------------|
| Taobao  |  48,749 |  39,493 | buy, cart, view          |
| JData   |  93,334 |  24,624 | buy, cart, collect, view |
| Tmall   |  41,738 |  11,953 | buy, cart, click, collect|



## Quick Start

Run with default hyperparameters on Taobao:

```bash
python src/main_bevae.py --dataset taobao --device cuda:0
```

This will train BE-VAE, early-stop on validation NDCG@10, save the best checkpoint to
`./checkpoint_bevae/taobao/model.pt`, then report test HR@{10,20,50,100} and NDCG@{10,20,50,100}.


## Training

Run train.sh to train BE-VAE for all datasets.

```bash
bash train.sh
```

---