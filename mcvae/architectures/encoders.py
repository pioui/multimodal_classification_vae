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
        self.fc_layer = nn.Linear(in_features=n_input, out_features=28*28)
        self.n_input = n_input
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
        x = x.view(n_batch, self.n_input)
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
            nn.Linear(in_features=n_input, out_features=512),            
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
            nn.Linear(in_features=n_input, out_features=1024),            
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
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None, kernel_size = 9,
    ):
        """  
        65 - 3x Conv 1D 32-128 ->
        trento: kernel_size = 9
        houston: kernel_size = 7
        """
        # TODO: describe architecture and choice for people 
        super().__init__() 
        self.fc_layer = nn.Linear(in_features=n_input, out_features=64)
       
        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=n_hidden, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)
        
        x = self.fc_layer(x) 

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
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None, kernel_size=9
    ):
        """  
        65 - 3x Conv 1D 128-512 ->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.fc_layer = nn.Linear(in_features=n_input, out_features=64)

        self.encoder_cv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=256, out_channels=n_hidden, kernel_size=kernel_size),
            nn.SELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout_rate),
        )


        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        x = self.fc_layer(x) 

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

class EncoderB5(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        p=13
        65,p,p -2x2D Conv 128-256->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=n_input, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        q = self.encoder_cv(x)
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

    
class BernoulliDecoderA5(nn.Module):
    """
    p=13
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = True,
    ):
        super().__init__()
        self.decoder_cv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_input, out_channels=128, kernel_size=5),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(in_channels=256, out_channels=n_output, kernel_size=5),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        n_samples, n_batch, n_latent = x.shape
        x = x.reshape(-1,n_latent) # 25*512, 10
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        means = self.decoder_cv(x)
        means = nn.Sigmoid()(means)
        _, channels, patch_size,_= means.shape
        means = means.reshape(n_samples, n_batch, channels, patch_size, patch_size )
        return means


class EncoderB6(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        """  
        p=5
        65,p,p -2x2D Conv 128-256->
        """
        # TODO: describe architecture and choice for people 
        super().__init__()        
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=n_input, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples=1, squeeze=True, reparam=True):
        n_batch = len(x)

        q = self.encoder_cv(x)
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

    
class BernoulliDecoderA6(nn.Module):
    """
    p=5
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = True,
    ):
        super().__init__()
        self.decoder_cv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_input, out_channels=128, kernel_size=3),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(in_channels=128, out_channels=n_output, kernel_size=3),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        n_samples, n_batch, n_latent = x.shape
        x = x.reshape(-1,n_latent) # 25*512, 10
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        means = self.decoder_cv(x)
        means = nn.Sigmoid()(means)
        _, channels, patch_size,_= means.shape
        means = means.reshape(n_samples, n_batch, channels, patch_size, patch_size )
        return means
if __name__ == "__main__":
    from torchsummary import summary

    # layer = EncoderB0(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))

    # layer = EncoderB1(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))

    # layer = EncoderB2(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))

    # layer = EncoderB2(n_input=65, n_output=20, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))

    # layer = EncoderB3(n_input=55, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,55))

    # layer = EncoderB4(n_input=2, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,2))

    # layer = EncoderB5(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (65,13,13))

    # layer = BernoulliDecoderA5(n_input=10, n_output=65, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # inx = torch.rand(25,5,10)
    # outx = layer(inx)
    # print(inx.shape, outx.shape)


    # layer = EncoderB6(n_input=65, n_output=10, n_hidden=128, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (65,5,5))

    # layer = BernoulliDecoderA6(n_input=10, n_output=65, dropout_rate=0.1, do_batch_norm=False)
    # inx = torch.rand(25,5,10)
    # outx = layer(inx)
    # print(inx.shape, outx.shape)

    # layer = EncoderB6(n_input=65, n_output=10, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))

    # layer = EncoderB7(n_input=65, n_output=20, n_hidden=512, dropout_rate=0.1, do_batch_norm=False)
    # summary(layer, (1,65))