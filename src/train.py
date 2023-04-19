# train.py
#     train model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import f1_score
import wandb


@torch.no_grad()
def evaluate(model, train_data, val_data, block_size, batch_size, batch_fn, prev_losses):
    model.eval()
    losses = {}
    f1s = {}
    for split, data in [('train', train_data), ('valid', val_data)]:
        losses[split] = []
        f1s[split] = []
        for _ in range(10):
            xs, ys = batch_fn(data, block_size, batch_size)
            ys_hat, loss = model(xs, ys)
            f1 = f1_score(ys.flatten().numpy(), ys_hat.argmax(1).flatten().numpy(), average='weighted')
            losses[split].append(loss.item())
            f1s[split].append(f1)
        f1s[split] = np.mean(f1s[split])
        losses[split] = np.mean(losses[split])
    model.train()
    if prev_losses is not None and prev_losses['valid'] > losses['valid'] and prev_losses['train'] > losses['train']:
        torch.save(model.state_dict(), 'lm.pth')
    return losses, f1s


def train(model, train_data, valid_data, opt, conf, batch_fn):
    wandb.init(project='mawsa', entity='syrkis', config=conf)
    # experiment.add_pytorch_models({'model': model})
    for i in range(conf['n_iters']):
        xs, ys = batch_fn(train_data, conf['block_size'], conf['batch_size'])
        _, loss = model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % (conf['n_iters'] // 50) == 0:
            losses, f1s = evaluate(model, train_data, valid_data, conf['block_size'], conf['batch_size'], batch_fn, losses)
            wandb.log({'train_loss': losses['train'], 'valid_loss': losses['valid'], 'train_f1': f1s['train'], 'valid_f1': f1s['valid']})


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model