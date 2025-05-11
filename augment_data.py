import random as r
import string as s
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import load

MODEL_PATH = "augment_model.pth"
PREPROCESSOR_PATH = "api/models/feature_transformer.pkl"


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 input_shape: int,
                 latent_dim: int,
                 device: str = 'auto'):
        super(VariationalAutoencoder, self).__init__()

        self.activation = nn.SELU()
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(device)
        self.latent_space = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, latent_dim),
            self.activation
        ).to(self.device)

        self.latent_space_mu = nn.Linear(latent_dim,
                                         latent_dim).to(self.device)
        self.latent_space_var = nn.Linear(latent_dim,
                                          latent_dim).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            self.activation,
            nn.Linear(256, input_shape)
        ).to(self.device)

    def encode(self, X: torch.Tensor) -> tuple[torch.Tensor,
                                               torch.Tensor]:
        encoded = self.encoder(X)
        mu = self.latent_space_mu(encoded)
        log_var = self.latent_space_var(encoded)
        return (mu, log_var)

    def reparametize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        theta = nn.functional.softmax(z, dim=1)
        decoded = self.decoder(theta)
        return decoded

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        mu, log_var = self.encode(X)
        z = self.reparametize(mu, log_var)
        X_reconst = self.decode(z)
        return (X_reconst, mu, log_var)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z).detach()

    def generate_from_random(self, count: int) -> np.ndarray:
        z = torch.normal(std=1, mean=0,
                         size=(count, self.latent_space),
                         dtype=torch.float32)
        generated = self.generate(z)
        generated[:, 2:] = torch.round(generated[:, 2:])
        return generated.cpu().numpy()


def generate_ids(count: int) -> list[str]:
    return [
        f"{r.randint(0, 9)}{r.randint(0, 9)}{r.randint(0, 9)}{r.randint(0, 9)}"
        f"-{r.choice(s.ascii_uppercase)}{r.choice(s.ascii_uppercase)}"
        f"{r.choice(s.ascii_uppercase)}{r.choice(s.ascii_uppercase)}"
        for _ in range(count)
    ]


def generate_data(n_samples: int):
    feature_transformer = load(PREPROCESSOR_PATH)
    augment_model = VariationalAutoencoder(45, 48, device="cpu")
    augment_model.load_state_dict(torch.load(MODEL_PATH))
    cat_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]
    num_cols = [
        "MonthlyCharges",
        "TotalCharges"
    ]
    generated_data = augment_model.generate_from_random(n_samples)
    generated_df = pd.DataFrame(
        data=np.hstack(
            [feature_transformer.transformers_[0][1].inverse_transform(
                generated_data[:, :2]
             ),
             feature_transformer.transformers_[1][1].inverse_transform(
                 generated_data[:, 2:]
             )]
        ), columns=num_cols + cat_cols)
    generated_df = generated_df.fillna(value="non")
    generated_df["customerID"] = generate_ids(n_samples)
    generated_df.to_csv("data/generated_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("count", default=100, type=int)
    args = parser.parse_args()
    generate_data(n_samples=args.count)
