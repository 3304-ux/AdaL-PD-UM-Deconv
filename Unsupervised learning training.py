import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from DATA import load_data
from Model_Unsupervised import AdaLISTA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, epochs, lr, weight_decay, step_size, gamma, D):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    max_grad_norm = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma)
    loss_train = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (b_y, b_D) in enumerate(train_loader):
            b_y = b_y.to(device)
            b_D = b_D.to(device)

            x_hat = model(b_y, b_D)
            D = D.to(device)
            seismic_records = torch.matmul(D, x_hat)

            target = b_y
            mse_loss = 0.5 * torch.nn.functional.mse_loss(seismic_records, target)
            l1_loss = torch.norm(x_hat, p=1)
            A = A  # weight value
            B = B  # weight value
            loss = A * mse_loss + B * l1_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += loss.item()

        loss_train.append(train_loss / len(train_loader))
        scheduler.step()

    return model

def main():
    train_loader, D, train_inputs = load_data()

    n_features, n_atoms = D.shape
    model = AdaLISTA(n_features, n_atoms, max_iter=100, lambd=1.0)
    model = model.to(device)

    trained_model = train_model(
    )

if __name__ == "__main__":
    main()
