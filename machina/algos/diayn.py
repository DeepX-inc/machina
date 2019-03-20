import torch
from machina import loss_functional as lf


def train(discrim, optim_discrim, on_traj, discrim_batch_size, epc_per_itr, num_skill):
    discrim_losses = []
    for batch in on_traj.random_batch(discrim_batch_size, epc_per_itr):
        optim_discrim.zero_grad()
        loss = lf.cross_ent_diayn(discrim, batch, num_skill)
        loss.backward()
        optim_discrim.step()
        discrim_losses.append(loss.item())
    return discrim_losses
