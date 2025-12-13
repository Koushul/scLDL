import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LIBLE(nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None, use_mixup=False, mixup_alpha=1.0):
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
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

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
                
                # Mixup
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X_mix = lam * batch_X + (1 - lam) * batch_X[index, :]
                    batch_L_mix = lam * batch_L + (1 - lam) * batch_L[index, :]
                    
                    mean, logvar, L_hat, D_hat, g = self.forward(batch_X_mix)
                    
                    # KL Divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                    
                    # Reconstruction Loss L
                    rec_L_loss = torch.sum((batch_L_mix - L_hat)**2, dim=1).mean()
                    
                    # Reconstruction Loss D
                    g_sq = g.pow(2) + 1e-6
                    rec_D_loss = torch.sum((batch_L_mix - D_hat)**2 / g_sq + torch.log(g_sq), dim=1).mean()
                    
                    loss = rec_L_loss + self.alpha * kl_loss + self.beta * rec_D_loss
                else:
                    mean, logvar, L_hat, D_hat, g = self.forward(batch_X)
                    
                    # KL Divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                    
                    # Reconstruction Loss L
                    rec_L_loss = torch.sum((batch_L - L_hat)**2, dim=1).mean()
                    
                    # Reconstruction Loss D
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
    def __init__(self, n_features, n_outputs, n_hidden=64, alpha=1.0, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None, use_mixup=False, mixup_alpha=1.0):
        super(LEVI, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
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
                
                # Mixup
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X = lam * batch_X + (1 - lam) * batch_X[index, :]
                    batch_L = lam * batch_L + (1 - lam) * batch_L[index, :]
                
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









from torchvision.models import resnet18, ResNet18_Weights

class ResidualBlock(nn.Module):
    def __init__(self, n_hidden):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ImprovedLEVI(nn.Module):
    """
    ImprovedLEVI: LEVI with a powerful pretrained encoder (ResNet18 for images) 
    or a deep Residual MLP for 1D data.
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1.0, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='resnet', input_shape=None, use_mixup=False, mixup_alpha=1.0, manifold_reg=0.0, gamma=1.0):
        super(ImprovedLEVI, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_latent = n_latent # Latent dim
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.encoder_type = encoder_type
        self.manifold_reg = manifold_reg
        
        # --- Feature Extractor ---
        if encoder_type == 'resnet' and input_shape is not None and len(input_shape) >= 2:
            # Image Data
            weights = ResNet18_Weights.DEFAULT
            self.feature_extractor = resnet18(weights=weights)
            
            # Modify first layer for 1 channel if input is (1, H, W)
            if input_shape[0] == 1:
                original_conv1 = self.feature_extractor.conv1
                self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    self.feature_extractor.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
                    
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            self.is_image = True
        else:
            # 1D Data (e.g., scRNA-seq) - Use Residual MLP
            self.is_image = False
            self.feature_dim = 512 # Project to same dim as ResNet for consistency
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Linear(256, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU()
            )

        # --- Variational Encoder: [Feature(X), L] -> Z ---
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_dim + n_outputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        
        self.enc_mean = nn.Linear(n_hidden, n_latent)
        self.enc_logvar = nn.Linear(n_hidden, n_latent)
        
        # --- Decoder X: Z -> X ---
        if self.is_image:
            self.decoder_fc = nn.Linear(n_latent, 64 * 7 * 7)
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:
            # Decoder for 1D data
            self.decoder_x_1d = nn.Sequential(
                nn.Linear(n_latent, 256),
                nn.ReLU(),
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Linear(256, n_features) 
                # No Sigmoid/Softmax at end, assume normalized counts (e.g. log1p)
                # or use specific activation if data is in [0,1]
            )
        
        # --- Decoder L: Z -> L ---
        self.decoder_L = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )
        
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, l, transform=False):
        if self.is_image:
            # Reshape X for ResNet
            if x.dim() == 2 and self.input_shape is not None:
                x_in = x.view(-1, *self.input_shape)
            else:
                x_in = x
        else:
            x_in = x
            
        # 1. Extract Features
        h_x = self.feature_extractor(x_in)
        
        # 2. Encode to Z
        inputs = torch.cat((h_x, l), dim=1)
        h = self.encoder_fc(inputs)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        
        if transform:
            return mean
            
        z = self.reparameterize(mean, logvar)
        
        # 3. Decode X
        if self.is_image:
            h_dec = self.decoder_fc(z)
            h_dec = h_dec.view(-1, 64, 7, 7)
            X_hat = self.decoder_cnn(h_dec)
            X_hat = X_hat.view(x.size(0), -1)
        else:
            X_hat = self.decoder_x_1d(z)
        
        # 4. Decode L
        L_hat = self.decoder_L(z)
        
        return mean, logvar, z, X_hat, L_hat, h_x

    def _manifold_loss(self, features, pred_y):
        """
        Computes Manifold Regularization Loss.
        L_reg = sum_{i,j} S_{ij} ||P_i - P_j||^2
        where S_{ij} is similarity (RBF kernel) between feature vectors.
        """
        # Normalize features for stable kernel computation
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Compute Similarity Matrix S (RBF Kernel)
        # ||f_i - f_j||^2 = 2 - 2 * f_i . f_j (for normalized vectors)
        # S_ij = exp(-gamma * ||f_i - f_j||^2)
        sim_matrix = torch.mm(features_norm, features_norm.t())
        dist_matrix = 2 - 2 * sim_matrix
        # Gamma heuristic: 1.0 or 1/n_features
        gamma = 1.0 
        S = torch.exp(-gamma * dist_matrix)
        
        # Compute Pairwise Label Distance
        # ||P_i - P_j||^2
        # Efficiently: P_dist = ||P_i||^2 + ||P_j||^2 - 2 P_i P_j^T
        p_norm = (pred_y ** 2).sum(1).view(-1, 1)
        p_dist = p_norm + p_norm.t() - 2 * torch.mm(pred_y, pred_y.t())
        
        # Loss = sum(S * P_dist)
        loss = torch.sum(S * p_dist)
        
        # Normalize by batch size squared
        return loss / (features.size(0) ** 2)

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.history = {'loss': [], 'manifold_loss': []}
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_manifold_loss = 0
            num_batches = 0
            
            for batch_X, batch_L in dataloader:
                # Mixup
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X_mix = lam * batch_X + (1 - lam) * batch_X[index, :]
                    batch_L_mix = lam * batch_L + (1 - lam) * batch_L[index, :]
                    
                    mean, logvar, z, X_hat, L_hat, h_x = self.forward(batch_X_mix, batch_L_mix)
                    
                    # For manifold loss, use detached features to avoid trivial solution (pushing features apart)
                    features_for_reg = h_x.detach()
                    
                    # KL Divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                    
                    if self.is_image:
                         batch_X_mix_flat = batch_X_mix.view(batch_X_mix.size(0), -1)
                    else:
                         batch_X_mix_flat = batch_X_mix

                    # Reconstruction X (MSE)
                    rec_X_loss = torch.sum((batch_X_mix_flat - X_hat)**2, dim=1).mean()
                    
                    # Reconstruction L (BCE/MSE)
                    rec_L_loss = F.binary_cross_entropy_with_logits(L_hat, batch_L_mix, reduction='sum') / batch_X.size(0)
                    
                    loss = rec_X_loss + self.gamma * rec_L_loss + self.alpha * kl_loss
                    
                else:
                    mean, logvar, z, X_hat, L_hat, h_x = self.forward(batch_X, batch_L)
                    # Use extracted features (detached) for manifold regularization anchor
                    features_for_reg = h_x.detach()
                    
                    # KL Divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                    
                    if self.is_image:
                         batch_X_flat = batch_X.view(batch_X.size(0), -1)
                    else:
                         batch_X_flat = batch_X

                    # Reconstruction X (MSE)
                    rec_X_loss = torch.sum((batch_X_flat - X_hat)**2, dim=1).mean()
                    
                    # Reconstruction L (BCE/MSE)
                    rec_L_loss = F.binary_cross_entropy_with_logits(L_hat, batch_L, reduction='sum') / batch_X.size(0)
                    
                    loss = rec_X_loss + self.gamma * rec_L_loss + self.alpha * kl_loss
                
                # Manifold Regularization
                if self.manifold_reg > 0:
                    # Use predicted probabilities for P
                    pred_probs = F.softmax(L_hat, dim=1)
                    m_loss = self._manifold_loss(features_for_reg, pred_probs)
                    loss += self.manifold_reg * m_loss
                    total_manifold_loss += m_loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_m_loss = total_manifold_loss / num_batches if num_batches > 0 else 0
            
            self.history['loss'].append(avg_loss)
            self.history['manifold_loss'].append(avg_m_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Manifold Loss: {avg_m_loss:.6f}")
                
        return self

    def predict(self, X, L):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            L_tensor = torch.FloatTensor(L).to(self.device)
            mean = self.forward(X_tensor, L_tensor, transform=True)
            L_hat = self.decoder_L(mean)
            return F.softmax(L_hat, dim=1).cpu().numpy()


class ConcentrationLE(nn.Module):
    """
    Concentration Label Enhancement (CDLLD).
    Based on Concentration Distribution Learning from Label Distributions.
    Models the target as a Dirichlet distribution derived from evidence.
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='mlp', input_shape=None, lambda_epochs=1, use_mixup=False, mixup_alpha=1.0):
        super(ConcentrationLE, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.lambda_epochs = lambda_epochs
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Encoder
        if encoder_type == 'resnet' and input_shape is not None and len(input_shape) >= 2:
             # Image Data
            weights = ResNet18_Weights.DEFAULT
            self.feature_extractor = resnet18(weights=weights)
            
            # Modify first layer for 1 channel if input is (1, H, W)
            if input_shape[0] == 1:
                original_conv1 = self.feature_extractor.conv1
                self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    self.feature_extractor.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
                    
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            
            # Head for Evidence
            self.encoder_head = nn.Sequential(
                nn.Linear(self.feature_dim, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs),
                nn.Softplus()
            )
            self.is_resnet = True
        elif encoder_type == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN encoder")
            
            # Simple CNN architecture (Same as LIBLE/LEVI for fair comparison)
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
                flattened_size = dummy_output.shape[1]
                
            self.encoder_fc = nn.Sequential(
                nn.Linear(flattened_size, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs),
                nn.Softplus() # Evidence must be non-negative
            )
            self.is_resnet = False
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), # Extra layer to match artifact Classifier slightly better
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs),
                nn.Softplus() # Evidence must be non-negative
            )
            self.is_resnet = False
            
        self.to(self.device)

    def forward(self, x):
        if hasattr(self, 'is_resnet') and self.is_resnet:
             # Reshape X if needed
            if x.dim() == 2 and self.input_shape is not None:
                x_in = x.view(-1, *self.input_shape)
            else:
                x_in = x
            h = self.feature_extractor(x_in)
            evidence = self.encoder_head(h)
        elif self.encoder_type == 'cnn':
            if x.dim() == 2 and self.input_shape is not None:
                x = x.view(-1, *self.input_shape)
            h = self.encoder_cnn(x)
            evidence = self.encoder_fc(h)
        else:
            evidence = self.encoder(x)
            
        alpha = evidence + 1
        return evidence, alpha

    def mse_loss(self, y, alpha, global_step=None):
        """
        MSE Loss for CDLLD.
        y: target logical labels (or distribution)
        alpha: concentration parameters
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        # m = expected probability
        m = alpha / S 
        
        # A: Squared Error between label and expected probability
        A = torch.sum((y - m) ** 2, dim=1, keepdim=True)
        
        # B: Variance term
        # Var(p) = alpha * (S - alpha) / (S^2 * (S + 1))
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        
        # Annealing (if implemented, from artifact: mse_loss(y, alpha, c, global_step, lambda_epochs))
        # Artifact doesn't seem to use global_step in the simplified code, but refers to it.
        # "loss += mse_loss(y, alpha...)"
        
        return torch.mean(A + B)

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.history = {'loss': []}
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_L in dataloader:
                
                # Mixup
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X = lam * batch_X + (1 - lam) * batch_X[index, :]
                    batch_L = lam * batch_L + (1 - lam) * batch_L[index, :]
                    
                optimizer.zero_grad()
                
                evidence, alpha = self.forward(batch_X)
                
                # Input L is treated as the target distribution y (logical or soft)
                loss = self.mse_loss(batch_L, alpha)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
                
        return self

    def predict(self, X):
        """
        Returns the belief mass distribution + uncertainty (background term).
        Output shape: (N, K+1) where the last column is uncertainty u.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            evidence, alpha = self.forward(X_tensor)
            S = torch.sum(alpha, dim=1, keepdim=True)
            
            # Belief masses b_k = alpha_k / S  <-- Wait, alpha = e + 1. 
            # Artifact: b = Yhat / S where Yhat is EVIDENCE? 
            # In artifact test(): Yhat = evidences[0].
            # S = sum(Yhat + 1).
            # b = Yhat / S.  (So b = e / S, not alpha / S)
            # u = classes / S.
            # Indeed: sum(b) + u = (sum(e) + K) / S = (S - K + K) / S = 1? 
            # S = sum(e) + K. 
            # sum(b) = sum(e)/S. u = K/S.
            # sum(b) + u = (sum(e) + K)/S = S/S = 1.
            # So b = evidence / S.
            
            b = evidence / S
            u = self.n_outputs / S
            
            return torch.cat((b, u), dim=1).cpu().numpy()


class HybridLEVI(nn.Module):
    """
    HybridLEVI: VAE-based Concentration Learning.
    Encoder: X -> Z (Latent)
    Decoder 1: Z -> X_hat (Reconstruction)
    Decoder 2: Z -> Evidence -> Alpha (Concentration)
    
    This learns latent features Z that are good for both reconstructing X and predicting labels.
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1.0, lr=1e-3, epochs=100, batch_size=32, device=None, encoder_type='cnn', input_shape=None, use_mixup=False, mixup_alpha=1.0, gamma=1.0):
        super(HybridLEVI, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.alpha = alpha # KL Weight
        self.gamma = gamma # Concentration Loss Weight
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Encoder: X -> Z
        if encoder_type == 'cnn':
            if input_shape is None: raise ValueError("input_shape required")
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                flat_size = self.encoder_cnn(dummy).shape[1]
            
            self.encoder_fc = nn.Sequential(
                nn.Linear(flat_size, n_hidden), nn.ReLU(),
                nn.Linear(n_hidden, n_hidden)
            )
        else:
            self.encoder_fc = nn.Sequential(
                nn.Linear(n_features, n_hidden), nn.ReLU(),
                nn.Linear(n_hidden, n_hidden)
            )
            
        self.enc_mean = nn.Linear(n_hidden, n_latent)
        self.enc_logvar = nn.Linear(n_hidden, n_latent)
        
        # Decoder X: Z -> X
        self.decoder_X = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        
        # Decoder Evidence: Z -> Evidence
        self.decoder_evidence = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
            nn.Softplus()
        )
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        if self.encoder_type == 'cnn':
            if x.dim() == 2 and self.input_shape: x = x.view(-1, *self.input_shape)
            h = self.encoder_cnn(x)
            h = self.encoder_fc(h)
        else:
            h = self.encoder_fc(x)
            
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        z = self.reparameterize(mean, logvar)
        
        x_hat = self.decoder_X(z)
        evidence = self.decoder_evidence(z)
        
        return mean, logvar, z, x_hat, evidence

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.history = {'loss': []}
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            for batch_X, batch_L in dataloader:
                
                # Mixup
                if self.use_mixup:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X = lam * batch_X + (1 - lam) * batch_X[index, :]
                    batch_L = lam * batch_L + (1 - lam) * batch_L[index, :]
                
                optimizer.zero_grad()
                mean, logvar, z, X_hat, evidence = self.forward(batch_X)
                
                # Losses
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                
                # Flatten batch_X for reconstruction loss
                batch_X_flat = batch_X.view(batch_X.size(0), -1)
                # Use sum over features (dim 1) to match VAE scale and balance KL
                rec_loss = torch.sum((batch_X_flat - X_hat)**2, dim=1).mean()
                
                # CDL Loss
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                m = alpha / S
                A = torch.sum((batch_L - m) ** 2, dim=1, keepdim=True)
                B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
                cdl_loss = torch.mean(A + B)
                
                loss = rec_loss + self.gamma * cdl_loss + self.alpha * kl_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return self

    def predict(self, X, L=None):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, _, _, _, evidence = self.forward(X_tensor)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            b = evidence / S
            u = self.n_outputs / S
            return torch.cat((b, u), dim=1).cpu().numpy()


# Import diffusion components
try:
    from .diffusion_classifier.card_model import ConditionalModel
    from .diffusion_classifier.diffusion_utils import make_beta_schedule, q_sample, p_sample_loop
except ImportError:
    # Fallback for when running script directly not as package
    try:
        from src.scLDL.diffusion_classifier.card_model import ConditionalModel
        from src.scLDL.diffusion_classifier.diffusion_utils import make_beta_schedule, q_sample, p_sample_loop
    except ImportError:
        pass # Will fail later if class is used

class DiffLEVI(nn.Module):
    """
    DiffLEVI: Label Enhancement via Diffusion (CARD-based).
    Replaces the VAE in LEVI with a Conditional Diffusion Model p(y|x).
    """
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, 
                 timesteps=1000, beta_schedule='linear', lr=1e-3, epochs=100, 
                 batch_size=32, device=None, encoder_type='mlp', input_shape=None):
        super(DiffLEVI, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.timesteps = timesteps
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        
        # --- Feature Extractor f(x) ---
        # We reuse the logic from ImprovedLEVI for consistency
        if encoder_type == 'resnet' and input_shape is not None and len(input_shape) >= 2:
            # Image Data
            weights = ResNet18_Weights.DEFAULT
            self.feature_extractor = resnet18(weights=weights)
            # Modify first layer for 1 channel if input is (1, H, W)
            if input_shape[0] == 1:
                original_conv1 = self.feature_extractor.conv1
                self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    self.feature_extractor.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            self.is_image = True
        else:
            # 1D Data (e.g., scRNA-seq)
            self.is_image = False
            self.feature_dim = n_latent # We project to latent dim directly for diffusion conditioning
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, self.feature_dim)
            )

        # --- Diffusion Model p(y_t-1 | y_t, x) ---
        # The ConditionalModel takes x (feature), y (noisy label), and t
        # We wrap it to adapt inputs
        
        # Beta schedule
        self.betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=timesteps).to(self.device).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        
        # Conditional Model
        self.diffusion_model = ConditionalModel(
            n_steps=timesteps + 1,
            data_dim=self.feature_dim, # Input will be f(x)
            y_dim=n_outputs,
            arch='linear', 
            feature_dim=n_latent, # Hidden dim for diffusion
            hidden_dim=n_hidden,
            guidance=False # We condition on x, which is standard in ConditionalModel
        )
        
        self.to(self.device)
        
    def fit(self, X, L, X_val=None, y_val=None):
        # Local import
        try:
            import enlighten
            use_enlighten = True
        except ImportError:
            use_enlighten = False

        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.history = {'loss': [], 'val_acc': []}
        
        self.train()
        
        if use_enlighten:
            manager = enlighten.get_manager()
            epoch_pbar = manager.counter(total=self.epochs, desc='Epochs', unit='epoch', color='green')
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Progress bar for batches
            if use_enlighten:
                batch_pbar = manager.counter(total=len(dataloader), desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch', color='blue', leave=False)
            
            self.train() # Ensure train mode
            for batch_X, batch_L in dataloader:
                optimizer.zero_grad()
                
                # 1. Extract features f(x)
                if self.is_image:
                    if batch_X.dim() == 2 and self.input_shape is not None:
                         x_in = batch_X.view(-1, *self.input_shape)
                    else:
                         x_in = batch_X
                    features = self.feature_extractor(x_in)
                else:
                    features = self.feature_extractor(batch_X)
                
                # 2. Diffusion forward pass (add noise to labels)
                batch_size = batch_X.size(0)
                t = torch.randint(low=0, high=self.timesteps, size=(batch_size // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.timesteps - 1 - t], dim=0)[:batch_size]
                
                noise = torch.randn_like(batch_L).to(self.device)
                
                # In CARD, y_0_hat is often used as mean for forward. 
                # Here we assume standard forward q(y_t|y_0) which means y_0_hat = y_0 (target labels)
                y_t = q_sample(batch_L, batch_L, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=noise)
                
                # 3. Predict noise
                output = self.diffusion_model(features, y_t, t)
                
                # 4. Loss
                loss = (noise - output).square().mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if use_enlighten:
                    batch_pbar.update()
                    # We can't easily change description of active counter in enlighten like tqdm set_postfix
                    # but likely sufficient to see progress. We can print or update separate status.
                    # Or use batch_pbar.desc = ... 
                    batch_pbar.desc = f"Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.4f}"
                
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.history['loss'].append(avg_loss)
            
            # Validation
            val_acc_str = ""
            if X_val is not None and y_val is not None:
                # Evaluate on validation set
                # Note: This might be slow for diffusion models
                try:
                    probs = self.predict(X_val) # predict sets eval()
                    y_pred = np.argmax(probs, axis=1)
                    val_acc = (y_pred == y_val).mean()
                    self.history['val_acc'].append(val_acc)
                    val_acc_str = f", Val Acc: {val_acc*100:.2f}%"
                except Exception as e:
                    print(f"Validation failed: {e}")
                    self.history['val_acc'].append(0)
            
            if use_enlighten:
                batch_pbar.close()
                epoch_pbar.update()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}{val_acc_str}")
            
            self.train() # Set back to train mode for next epoch
            
        if use_enlighten:
            manager.stop()
                
        return self
        
    def predict(self, X, n_samples=1):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            batch_size = X_tensor.size(0)
            
            # 1. Extract features
            if self.is_image:
                 if X_tensor.dim() == 2 and self.input_shape is not None:
                     x_in = X_tensor.view(-1, *self.input_shape)
                 else:
                     x_in = X_tensor
                 features = self.feature_extractor(x_in)
            else:
                 features = self.feature_extractor(X_tensor)
                 
            # 2. Denoise loop
            if n_samples > 1:
                features = features.repeat_interleave(n_samples, dim=0)

            y_0_hat = torch.ones(features.size(0), self.n_outputs).to(self.device) / self.n_outputs
            y_T_mean = y_0_hat # Simple prior
            
            y_0_pred = p_sample_loop(
                self.diffusion_model, 
                features, 
                y_0_hat, 
                y_T_mean, 
                self.timesteps, 
                self.alphas, 
                self.one_minus_alphas_bar_sqrt, 
                only_last_sample=True
            )
            
            if n_samples > 1:
                 y_0_pred = y_0_pred.reshape(batch_size, n_samples, self.n_outputs).mean(dim=1)
            
            return F.softmax(y_0_pred, dim=1).cpu().numpy()
            
    def get_latent(self, X):
        """Returns the feature embedding f(x)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.is_image:
                 if X_tensor.dim() == 2 and self.input_shape is not None:
                     x_in = X_tensor.view(-1, *self.input_shape)
                 else:
                     x_in = X_tensor
                 features = self.feature_extractor(x_in)
            else:
                 features = self.feature_extractor(X_tensor)
            return features.cpu().numpy()
