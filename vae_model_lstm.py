import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, latent_size=128, num_layers=1):
        super(MolecularVAE_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # --- 1. ENCODER ---
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)       # Media μ
        self.fc_logvar = nn.Linear(hidden_size, latent_size)   # Log-varianza σ²
        
        # --- 2. DECODER ---
        self.decoder_input_h = nn.Linear(latent_size, hidden_size)  # z → h₀ del decoder
        self.decoder_input_c = nn.Linear(latent_size, hidden_size)  # z → c₀ del decoder
        self.decoder_rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)            # Logits sobre vocabulario
        
    def reparameterize(self, mu, logvar):
        """Truco de reparametrización: z = μ + σ * ε, con ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # x: [batch, seq_len] con formato [SOS, t1, t2, ..., EOS, PAD, PAD, ...]
        
        # --- ENCODING ---
        embed = self.embedding(x)
        _, (hidden, cell) = self.encoder_rnn(embed)
        h = hidden[-1]  # Tomamos el estado oculto de la última capa
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Prevención de overflow que produce Inf/NaN y corrompe CuDNN
        logvar = torch.clamp(logvar, min=-20, max=20)
        mu = torch.clamp(mu, min=-20, max=20)
        
        z = self.reparameterize(mu, logvar)
        
        # --- DECODING (Teacher Forcing) ---
        # Input del decoder: x[:, :-1] = [SOS, t1, ..., tn-1]
        # Target esperado:   x[:, 1:]  = [t1, t2, ..., EOS, PAD...]
        # Repetimos h_0 y c_0 para cada capa del decoder
        h_decoder = self.decoder_input_h(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_decoder = self.decoder_input_c(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        embed_decoder = self.embedding(x[:, :-1])          # Embeddings shifted
        out, _ = self.decoder_rnn(embed_decoder, (h_decoder, c_decoder))
        prediction = self.fc_out(out)  # [batch, seq_len-1, vocab_size]
        
        return prediction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, kl_weight, pad_idx=0):
    """
    Loss del VAE: L = Recon + β·KL
    
    Args:
        recon_x: Logits del decoder [batch, seq_len-1, vocab_size]
        x: Secuencia original [batch, seq_len] → [SOS, t1, ..., EOS, PAD...]
        mu, logvar: Parámetros de la distribución latente q(z|x)
        kl_weight: Factor β para KL annealing
        pad_idx: Índice del token de padding (ignorado en cross-entropy)
    
    Returns:
        (loss_total, recon_loss, kl_loss)
    """
    vocab_size = recon_x.size(-1)
    target = x[:, 1:]  # Target sin SOS: [t1, t2, ..., EOS, PAD...]
    
    # Reconstrucción: Cross-Entropy sumada sobre todos los tokens no-padding
    recon_loss = F.cross_entropy(
        recon_x.reshape(-1, vocab_size), 
        target.reshape(-1), 
        ignore_index=pad_idx, 
        reduction='sum'
    )
    
    # KL Divergence: D_KL(q(z|x) || p(z)), con p(z) = N(0, I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (kl_weight * kld), recon_loss, kld
