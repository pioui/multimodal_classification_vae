import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as db

class EncoderB0(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 -FC-> 28x28 -3x2D Conv 32-128->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()
        # FOR TRENTO
        self.fc_layer = nn.Linear(in_features=65, out_features=28*28)
        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)


        # FOR TRENTO
        x = x.view(n_batch, 65)
        x = self.fc_layer(x)

        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )


class EncoderB1(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 3x FC 512-128 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Linear(in_features=65, out_features=512),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=256),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=n_hidden),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, -1)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )

class EncoderB2(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 3x FC 1024-256 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Linear(in_features=65, out_features=1024),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=1024, out_features=512),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=n_hidden),            
            nn.SELU(),
            nn.Dropout(p=dropout_rate),            
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, -1)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )

class EncoderB3(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 3x Conv 1D 32-128 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=n_hidden, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, 1, -1)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )

class EncoderB4(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        65 - 5x Conv 1D 128-512 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=256, out_channels=n_hidden, kernel_size=9),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )


        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x_reshape = x.view(n_batch, 1, -1)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )


if __name__ == "__main__":
    from torchsummary import summary

    layer = EncoderB0(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB1(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB2(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB2(n_input=65, n_output=20, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB3(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))


    layer = EncoderB4(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB5(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB6(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))

    layer = EncoderB7(n_input=65, n_output=20, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    summary(layer, (1,65))