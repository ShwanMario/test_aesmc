from aesmc import losses
from aesmc import statistics

import itertools
import sys
import torch.nn as nn
import torch.utils.data

def get_chained_params(*objects):
    result = []
    for object in objects:
        if (object is not None) and isinstance(object, nn.Module):
            result = itertools.chain(result, object.parameters())

    if isinstance(result, list):
        return None
    else:
        return result


def train(dataloader, num_particles, algorithm, initial, transition, emission,
          proposal, num_epochs, num_iterations_per_epoch=None,
          optimizer_algorithm=torch.optim.Adam, optimizer_kwargs={},
          callback=None, args=None):
    device = args.device
    parameters = get_chained_params(initial, transition, emission, proposal)
    optimizer = optimizer_algorithm(parameters, **optimizer_kwargs)
    for epoch_idx in range(num_epochs):
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader):
            true_latents = latents_and_observations[0]
            true_latents = [true_latent.to(device).unsqueeze(-1) for true_latent in true_latents]
            observations = latents_and_observations[1]
            observations = [observation.to(device) for observation in observations]
            if num_iterations_per_epoch is not None:
                if epoch_iteration_idx == num_iterations_per_epoch:
                    break
            optimizer.zero_grad()
            loss = losses.get_loss(observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents)
            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(epoch_idx, epoch_iteration_idx, loss, initial,
                         transition, emission, proposal)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def __getitem__(self, index):
        # TODO this is wrong, obs can be dict
        latents_and_observations = statistics.sample_from_prior(self.initial, self.transition,
                                     self.emission, self.num_timesteps,
                                     self.batch_size)
        return [list(map(lambda latent: latent.detach().squeeze(0), latents_and_observations[0])),
                list(map(lambda observation: observation.detach().squeeze(0), latents_and_observations[1]))]

    def __len__(self):
        return sys.maxsize  # effectively infinite


def get_synthetic_dataloader(initial, transition, emission, num_timesteps,
                             batch_size):
    return torch.utils.data.DataLoader(
        SyntheticDataset(initial, transition, emission, num_timesteps,
                         batch_size),
        batch_size=1,
        collate_fn=lambda x: x[0])
