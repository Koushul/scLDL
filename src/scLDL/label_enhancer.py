import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LIBLE(nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None):
        super(LIBLE, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.mps.is_available():
            self.device = torch.device("mps")
        self.encoder_type = encoder_type
        self.input_shape = input_shape

        # Encoder
        if encoder_type == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN encoder")
            
            # Simple CNN architecture for MNIST-like data (1 channel)
            # Input: (C, H, W)
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # -> (32, H/2, W/2)
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # -> (64, H/4, W/4)
                nn.Flatten()
            )
            
            # Calculate flattened size
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape)
                dummy_output = self.encoder_cnn(dummy_input)
                flattened_size = dummy_output.shape[1]
                
            self.encoder_fc = nn.Sequential(
                nn.Linear(flattened_size, n_hidden),
                nn.Tanh()
            )
        else:
            self.encoder_hidden = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.Tanh()
            )
            
        self.encoder_mean = nn.Linear(n_hidden, n_latent)
        self.encoder_logvar = nn.Linear(n_hidden, n_latent) # Output log variance directly

        # Decoder L (Label reconstruction)
        self.decoder_L = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs)
        )

        # Decoder D (Label distribution reconstruction)
        self.decoder_D = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs)
        )

        # Decoder g (Uncertainty/Scale)
        self.decoder_g = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)

    def encode(self, x):
        if self.encoder_type == 'cnn':
            # Reshape if input is flattened
            if x.dim() == 2 and self.input_shape is not None:
                x = x.view(-1, *self.input_shape)
            
            h = self.encoder_cnn(x)
            h = self.encoder_fc(h)
        else:
            h = self.encoder_hidden(x)
            
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, transform=False):
        mean, logvar = self.encode(x)
        if transform:
            # For prediction, we might use just the mean to generate D
            return self.decoder_D(mean)
        
        z = self.reparameterize(mean, logvar)
        
        L_hat = self.decoder_L(z)
        D_hat = self.decoder_D(z)
        g = self.decoder_g(z)
        
        return mean, logvar, L_hat, D_hat, g

    @torch.no_grad()
    def get_z(self, x):
        x_tensor = torch.FloatTensor(x).to(self.device)
        mean, logvar = self.encode(x_tensor)
        z = self.reparameterize(mean, logvar)
        return z.detach().cpu().numpy()

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_L in dataloader:
                optimizer.zero_grad()
                
                mean, logvar, L_hat, D_hat, g = self.forward(batch_X)
                
                # KL Divergence
                # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                
                # Reconstruction Loss L
                rec_L_loss = torch.sum((batch_L - L_hat)**2, dim=1).mean()
                
                # Reconstruction Loss D (Gaussian NLL-like)
                # Original: g**(-2) * (L - D_hat)**2 + log(g**2)
                # Note: g is sigmoid output (0, 1). g**2 might be small.
                # Adding epsilon for stability if needed, but following formula:
                g_sq = g.pow(2) + 1e-6
                rec_D_loss = torch.sum((batch_L - D_hat)**2 / g_sq + torch.log(g_sq), dim=1).mean()
                
                loss = rec_L_loss + self.alpha * kl_loss + self.beta * rec_D_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")
                
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.forward(X_tensor, transform=True)
            return F.softmax(logits, dim=1).cpu().numpy()

class LEVI(nn.Module):
    """
    LEVI: Label Enhancement via Variational Inference.
    Reimplemented using PyTorch.
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, alpha=1.0, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None):
        super(LEVI, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        
        # Encoder: [X, L] -> [mean, logvar]
        if encoder_type == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN encoder")
            
            # CNN for X
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
            
            # Calculate flattened size
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape)
                dummy_output = self.encoder_cnn(dummy_input)
                flattened_size = dummy_output.shape[1]
            
            # FC part: [CNN(X), L] -> hidden -> [mean, logvar]
            self.encoder_fc = nn.Sequential(
                nn.Linear(flattened_size + n_outputs, n_hidden),
                nn.Softplus(),
                nn.Linear(n_hidden, n_outputs * 2)
            )
        else:
            # MLP for [X, L]
            self.encoder = nn.Sequential(
                nn.Linear(n_features + n_outputs, n_hidden),
                nn.Softplus(),
                nn.Linear(n_hidden, n_outputs * 2) # Output mean and logvar
            )
        
        # Decoder X: samples -> X
        self.decoder_X = nn.Sequential(
            nn.Linear(n_outputs, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_features)
        )
        
        # Decoder L: samples -> L
        self.decoder_L = nn.Sequential(
            nn.Linear(n_outputs, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_outputs)
        )
        
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, X, L, transform=False):
        if self.encoder_type == 'cnn':
            # Reshape X if needed
            if X.dim() == 2 and self.input_shape is not None:
                X_reshaped = X.view(-1, *self.input_shape)
            else:
                X_reshaped = X
                
            h_x = self.encoder_cnn(X_reshaped)
            inputs = torch.cat((h_x, L), dim=1)
            latent = self.encoder_fc(inputs)
        else:
            inputs = torch.cat((X, L), dim=1)
            latent = self.encoder(inputs)
        
        mean = latent[:, :self.n_outputs]
        # We use logvar instead of softplus(var) for numerical stability in PyTorch VAEs usually.
        # But to match original: "var = tf.math.softplus(latent[:, self._n_outputs:])"
        # Let's stick to logvar for standard implementation unless strictly required.
        logvar = latent[:, self.n_outputs:]
        
        if transform:
            return mean
            
        z = self.reparameterize(mean, logvar)
        
        X_hat = self.decoder_X(z)
        L_hat = self.decoder_L(z)
        
        return mean, logvar, z, X_hat, L_hat

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_L in dataloader:
                optimizer.zero_grad()
                
                mean, logvar, z, X_hat, L_hat = self.forward(batch_X, batch_L)
                
                # KL Divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                
                # Reconstruction X (MSE)
                rec_X_loss = torch.mean(torch.mean((batch_X - X_hat)**2, dim=1))
                
                # Reconstruction L (BCE)
                rec_L_loss = F.binary_cross_entropy_with_logits(L_hat, batch_L, reduction='none')
                rec_L_loss = torch.mean(torch.mean(rec_L_loss, dim=1))
                
                # Main loss: sum((L - samples)**2)
                main_loss = torch.sum((batch_L - z)**2)
                
                # Recalculate component losses as sums to match original scale
                kl_sum = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                rec_X_sum = torch.sum((batch_X - X_hat)**2)
                rec_L_sum = F.binary_cross_entropy_with_logits(L_hat, batch_L, reduction='sum')
                
                loss = main_loss + self.alpha * (kl_sum + rec_X_sum + rec_L_sum)
                
                loss = loss / batch_X.shape[0]
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")
                
        return self

    def predict(self, X, L):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            L_tensor = torch.FloatTensor(L).to(self.device)
            mean = self.forward(X_tensor, L_tensor, transform=True)
            return F.softmax(mean, dim=1).cpu().numpy()






class HybridLE(nn.Module):
    """
    HybridLE: Combines LIBLE and LEVI using a Dual Encoder (CVAE-like) approach.
    - Encoder Prior (X -> Z): Used for prediction (Inductive).
    - Encoder Posterior (X, L -> Z): Used for training (Transductive guidance).
    - Decoders: Reconstructs D (Distribution), g (Uncertainty), X (Features), and L (Label).
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3, gamma=1e-3, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None):
        super(HybridLE, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.alpha = alpha # KL weight
        self.beta = beta   # D reconstruction weight
        self.gamma = gamma # X reconstruction weight
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.mps.is_available():
            self.device = torch.device("mps")
        self.encoder_type = encoder_type
        self.input_shape = input_shape

        # --- Encoder Prior: P(Z|X) ---
        if encoder_type == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN encoder")
            
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
            
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape)
                dummy_output = self.encoder_cnn(dummy_input)
                self.flattened_size = dummy_output.shape[1]
                
            self.encoder_prior_fc = nn.Sequential(
                nn.Linear(self.flattened_size, n_hidden),
                nn.ReLU()
            )
        else:
            self.encoder_prior_fc = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU()
            )
            
        self.prior_mean = nn.Linear(n_hidden, n_latent)
        self.prior_logvar = nn.Linear(n_hidden, n_latent)

        # --- Encoder Posterior: Q(Z|X, L) ---
        
        if encoder_type == 'cnn':
            input_dim_post = self.flattened_size + n_outputs
        else:
            input_dim_post = n_features + n_outputs
            
        self.encoder_post_fc = nn.Sequential(
            nn.Linear(input_dim_post, n_hidden),
            nn.ReLU()
        )
        self.post_mean = nn.Linear(n_hidden, n_latent)
        self.post_logvar = nn.Linear(n_hidden, n_latent)

        # --- Decoders ---
        
        # Decoder L (Label Reconstruction) - Strong supervision
        self.decoder_L = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )

        # Decoder D (Label Distribution) - From LIBLE
        self.decoder_D = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )
        
        # Decoder g (Uncertainty) - From LIBLE
        self.decoder_g = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        
        # Decoder X (Feature Reconstruction) - From LEVI
        self.decoder_X = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        
        self.to(self.device)

    def encode_prior(self, x):
        if self.encoder_type == 'cnn':
            if x.dim() == 2 and self.input_shape is not None:
                x = x.view(-1, *self.input_shape)
            h = self.encoder_cnn(x)
            h = self.encoder_prior_fc(h)
        else:
            h = self.encoder_prior_fc(x)
        return self.prior_mean(h), self.prior_logvar(h)

    def encode_post(self, x, l):
        if self.encoder_type == 'cnn':
            if x.dim() == 2 and self.input_shape is not None:
                x = x.view(-1, *self.input_shape)
            h_x = self.encoder_cnn(x) # Shared CNN
            inputs = torch.cat((h_x, l), dim=1)
        else:
            inputs = torch.cat((x, l), dim=1)
            
        h = self.encoder_post_fc(inputs)
        return self.post_mean(h), self.post_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, l=None, transform=False):
        # If transform=True, we are predicting (using Prior)
        if transform:
            mean, _ = self.encode_prior(x)
            return self.decoder_D(mean)
        
        # Training
        if l is None:
            raise ValueError("L must be provided for training forward pass")
            
        # 1. Posterior Encode (Teacher)
        post_mean, post_logvar = self.encode_post(x, l)
        z_post = self.reparameterize(post_mean, post_logvar)
        
        # 2. Prior Encode (Student)
        prior_mean, prior_logvar = self.encode_prior(x)
        
        # 3. Decode from Posterior Z
        L_hat = self.decoder_L(z_post)
        D_hat = self.decoder_D(z_post)
        g = self.decoder_g(z_post)
        X_hat = self.decoder_X(z_post)
        
        return prior_mean, prior_logvar, post_mean, post_logvar, z_post, L_hat, D_hat, g, X_hat

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_L in dataloader:
                optimizer.zero_grad()
                
                prior_mean, prior_logvar, post_mean, post_logvar, z_post, L_hat, D_hat, g, X_hat = self.forward(batch_X, batch_L)
                
                # KL Divergence: KL(Posterior || Prior)
                kl_element = (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp() \
                             - 1 + prior_logvar - post_logvar
                kl_loss = 0.5 * torch.sum(kl_element, dim=1).mean()
                
                # Prior Regularization KL(Prior || N(0,1))
                kl_prior_std = -0.5 * torch.sum(1 + prior_logvar - prior_mean.pow(2) - prior_logvar.exp(), dim=1).mean()
                
                # Reconstruction L (MSE) - Strong Supervision
                rec_L_loss = torch.sum((batch_L - L_hat)**2, dim=1).mean()

                # Reconstruction D (Weighted MSE with g) - LIBLE style
                g_sq = g.pow(2) + 1e-6
                rec_D_loss = torch.sum((batch_L - D_hat)**2 / g_sq + torch.log(g_sq), dim=1).mean()
                
                # Reconstruction X (MSE) - LEVI style
                batch_X_flat = batch_X.view(batch_X.size(0), -1)
                rec_X_loss = torch.sum((batch_X_flat - X_hat)**2, dim=1).mean()
                
                # Total Loss
                loss = rec_L_loss + self.beta * rec_D_loss + self.gamma * rec_X_loss + self.alpha * (kl_loss + 0.1 * kl_prior_std)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")
                
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.forward(X_tensor, transform=True)
            return F.softmax(logits, dim=1).cpu().numpy()

