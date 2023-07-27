import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======

        self.in_channels = in_channels
        self.out_channels = out_channels

        modules.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.Tanh())
        modules.append(nn.MaxPool2d(2))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules.append(nn.ConvTranspose2d(in_channels, 512, 4, stride=2, padding=1, output_padding=0))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(512, 256, 4,stride=2, padding=1, output_padding=0))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(256, 128, 4,stride=2, padding=1, output_padding=0))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(128, 64, 4,stride=2, padding=1, output_padding=0))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.Tanh())
        modules.append(nn.ConvTranspose2d(64, out_channels, 4,stride=2, padding=1, output_padding=0))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        # adding Gaussian MLP layers to represent encoder/decoder as presented in
        # Auto-Encoding Variational Bayes paper

        self.hidden_channel = self.features_encoder.out_channels # 1024
        
        # what is the size of the output images after all the convolutions? 2x2?
        # n_features = 4096
        # self.features_shape = torch.Size([1024, 2, 2])
        
        self.mu = nn.Linear(in_features=n_features, out_features=z_dim)    # 4096 -> 2
        self.sigma = nn.Linear(in_features=n_features, out_features=z_dim) # 4096 -> 2

        self.decode_MLP = nn.Linear(z_dim, n_features) # 2 -> 4096


        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x) # torch.Size([B, 1024, 2, 2])
        
        h_flat = torch.flatten(h, start_dim=1) # [B x 4096]
 
        mu = self.mu(h_flat)                   #  [B, 2]
        log_sigma2 = self.sigma(h_flat)
        I = torch.randn(log_sigma2.shape).to(h.device)
        z = log_sigma2*I + mu


        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        
        features_num, H, W = self.features_shape
        
        h = self.decode_MLP(z).reshape(-1, features_num, H, W ) # [B, 1024, 2, 2]
        
        # print(h.shape)
        
        # h = self.decode_MLP(z).reshape(-1, self.hidden_ch, 2, 2)
        x_rec = self.features_decoder(h)
        
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z = torch.randn(n, self.z_dim).to(device)
            samples = self.decode(z).cpu()
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    
    data_loss = torch.mean((x - xr) ** 2) / x_sigma2
    d_z = z_mu.shape[-1]
    mu_L2 = torch.sum(z_mu ** 2, dim=(-1))
    z_sigma2 = torch.exp(z_log_sigma2)
    Si_tr = torch.sum(z_sigma2, dim=(-1))
    Si_lgdt = torch.log(torch.prod(z_sigma2, dim=(-1)))
    kldiv_loss = torch.mean(Si_tr + mu_L2 - d_z - Si_lgdt)

    loss = data_loss + kldiv_loss
    
    # ========================

    return loss, data_loss, kldiv_loss
