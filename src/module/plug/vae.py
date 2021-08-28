import torch
import torch.nn as nn

def build_mlp(num_layers, in_dim, hidden_dim, out_dim):
    if num_layers == 4:
        return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
    elif num_layers == 3:
        return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
    elif num_layers == 2: 
        return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

class PlugVariationalAutoEncoder(nn.Module):
    def __init__(self, num_layers, code_dim, hidden_dim, latent_dim):
        super(PlugVariationalAutoEncoder, self).__init__()
        self.encoder = build_mlp(num_layers, code_dim, hidden_dim, latent_dim)
        self.decoder = build_mlp(num_layers, latent_dim + 1, hidden_dim, code_dim)
        self.linear_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.linear_logstd = torch.nn.Linear(latent_dim, latent_dim)

        self.latent_dim = latent_dim
    
    def forward(self, x, y):
        out = self.encoder(x)
        mu = self.linear_mu(out)
        std = (self.linear_logstd(out).exp() + 1e-6)
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        
        z = q.rsample()
        out = self.decoder(torch.cat([z, y.view(-1, 1)], dim=1))
        
        return out, z, p, q

    def decode(self, y):
        mu = torch.zeros(y.size(0), self.latent_dim, device=y.device)
        std = torch.ones(y.size(0), self.latent_dim, device=y.device)
        p = torch.distributions.Normal(mu, std)
        z = p.sample()
        x_hat = self.decoder(torch.cat([z, y], dim=1))
        if self.vq:
            x_hat = torch.argmax(x_hat.view(-1, self.vq_num_vocabs), dim=1).view(y.size(0), -1)
            
        return x_hat

class PlugDiscreteVariationalAutoEncoder(nn.Module):
    def __init__(self, vq, vq_num_vocabs, code_dim, latent_dim, hidden_dim, plug_beta):
        super(PlugVariationalAutoEncoder, self).__init__()
        self.vq = vq
        self.vq_num_vocabs = vq_num_vocabs
        code_dim = code_dim * vq_num_vocabs if vq else code_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(code_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )   
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+1, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, code_dim),
        )

        self.linear_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.linear_logstd = torch.nn.Linear(latent_dim, latent_dim)

        self.plug_beta = plug_beta
        self.latent_dim = latent_dim

    def step(self, x, y):
        loss, statistics = 0.0, dict()
        
        if self.vq:
            out = torch.nn.functional.one_hot(x, self.vq_num_vocabs).float().view(x.size(0), -1)   
        else:
            out = x

        out = self.encoder(out)
        mu = self.linear_mu(out)
        std = (self.linear_logstd(out).exp() + 1e-6)
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        
        z = q.rsample()
        x_hat = self.decoder(torch.cat([z, y.view(-1, 1)], dim=1))
        
        if self.vq:
            x_hat = x_hat.view(-1, self.vq_num_vocabs)
            recon_loss = torch.nn.functional.cross_entropy(x_hat, x.view(-1), reduction="mean")
            statistics["acc/plug"] = (torch.argmax(x_hat, dim=-1) == x.view(-1)).float().mean()
        else:
            recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl_loss = log_qz - log_pz
        kl_loss = kl_loss.mean()
        loss = self.plug_beta * kl_loss + recon_loss

        statistics["loss/plug/recon"] = recon_loss
        statistics["loss/plug/kl"] = kl_loss

        return loss, statistics

    def sample(self, y):
        mu = torch.zeros(y.size(0), self.latent_dim, device=y.device)
        std = torch.ones(y.size(0), self.latent_dim, device=y.device)
        p = torch.distributions.Normal(mu, std)
        z = p.sample()
        x_hat = self.decoder(torch.cat([z, y], dim=1))
        if self.vq:
            x_hat = torch.argmax(x_hat.view(-1, self.vq_num_vocabs), dim=1).view(y.size(0), -1)
            
        return x_hat
