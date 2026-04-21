import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVAE(nn.Module):
    _ACTIVATIONS = {
        'tanh': torch.tanh,
        'relu': F.relu,
        'gelu': F.gelu,
        'sigmoid': torch.sigmoid,
        'elu': F.elu,
        'leaky_relu': F.leaky_relu,
    }

    def __init__(self, n_items, hidden_dims, latent_dim, input_matrix,
                 dropout=0.0, activation='tanh', normalize_input=True, base_rate=None):
        """
        Args:
            hidden_dims:    list of hidden layer sizes, e.g. [1024, 512].
                            Encoder: input → hidden_dims[0] → ... → hidden_dims[-1] → latent
                            Decoder: latent → hidden_dims[-1] → ... → hidden_dims[0] → input
            input_matrix:   user-item weighted matrix (buy + weighted auxiliary behaviors).
                            Used as both the encoder input and the reconstruction target.
            base_rate:      optional per-user base-rate vector added to zero positions of the
                            reconstruction target.
        """
        super(BEVAE, self).__init__()
        self.n_items = n_items
        input_dim = n_items + 1

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        enc_dims = [input_dim] + hidden_dims
        self.encoder_layers = nn.ModuleList(
            nn.Linear(enc_dims[i], enc_dims[i + 1]) for i in range(len(enc_dims) - 1)
        )
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder_layers = nn.ModuleList(
            nn.Linear(dec_dims[i], dec_dims[i + 1]) for i in range(len(dec_dims) - 1)
        )

        self.dropout = nn.Dropout(dropout)
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_fn = self._ACTIVATIONS[activation]
        self.normalize_input = normalize_input

        self.register_buffer('input_matrix', input_matrix)
        if base_rate is not None:
            self.register_buffer('base_rate', base_rate)
        else:
            self.base_rate = None

        self.reset_parameters()

    def reset_parameters(self):
        layers = (list(self.encoder_layers) + [self.mu_head, self.logvar_head]
                  + list(self.decoder_layers))
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        h = F.normalize(x, p=2, dim=1) if self.normalize_input else x
        h = self.dropout(h)
        for layer in self.encoder_layers:
            h = self.activation_fn(layer(h))
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        h = z
        for layer in self.decoder_layers[:-1]:
            h = self.activation_fn(layer(h))
        return self.decoder_layers[-1](h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def _get_rows(self, user_ids):
        x = self.input_matrix.index_select(0, user_ids)
        if x.is_sparse:
            x = x.to_dense()
        return x

    def _apply_baserate(self, x, user_ids):
        br = self.base_rate[user_ids].unsqueeze(1)
        zero_mask = (x == 0).float()
        return x + zero_mask * br

    def loss(self, user_ids, beta=1.0):
        x = self._get_rows(user_ids)
        x_tgt = self._apply_baserate(x, user_ids) if self.base_rate is not None else x

        logits, mu, logvar = self.forward(x)

        log_softmax = F.log_softmax(logits, dim=1)
        recon_loss = -torch.mean(torch.sum(log_softmax * x_tgt, dim=1))
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def predict(self, user_indices):
        x = self._get_rows(user_indices.long())
        logits, _, _ = self.forward(x)
        return logits
