from . import math
from . import state
from resamplers.resamplers import resampler
import numpy as np
import torch
import copy
import sys
sys.path.append('../test/')

from test.models import lgssm

def infer(inference_algorithm, observations, initial, transition, emission,
          proposal, num_particles, return_log_marginal_likelihood=False,
          return_latents=False, return_original_latents=True,
          return_log_weight=True, return_log_weights=True,
          return_ancestral_indices=False, args=None, true_latents=None):
    """Perform inference on a state space model using either sequential Monte
    Carlo or importance sampling.

    Args:
        inference_algorithm: is or smc (string)
        observations: list of tensors [batch_size, ...] or
            dicts thereof of length num_timesteps
        initial: a callable object (function or nn.Module) which has no
            arguments and returns a torch.distributions.Distribution or a dict
            thereof
        transition: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        emission: a callable object (function or nn.Module) with signature:
            Args:
                true_latents: list of length (time + 1) where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        proposal: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                observations: list of length num_timesteps where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        num_particles: int; number of particles
        return_log_marginal_likelihood: bool (default: False)
        return_latents: bool (default: True)
        return_original_latents: bool (default: False); only applicable for smc
        return_log_weight: bool (default: True)
        return_log_weights: bool (default: False)
        return_ancestral_indices: bool (default: False); only applicable for
            smc
    Returns:
        a dict containing key-value pairs for a subset of the following keys
        as specified by the return_{} parameters:
            log_marginal_likelihood: tensor [batch_size]
            latents: list of tensors (or dict thereof)
                [batch_size, num_particles, ...] of length len(observations)
            original_latents: list of tensors (or dict thereof)
                [batch_size, num_particles, ...] of length len(observations)
            log_weight: tensor [batch_size, num_particles]
            log_weights: list of tensors [batch_size, num_particles]
                of length len(observations)
            ancestral_indices: list of `torch.LongTensor`s
                [batch_size, num_particles] of length len(observations)

        Note that (latents, log_weight) characterize the posterior.
    """
    device = args.device
    if not (inference_algorithm == 'is' or inference_algorithm == 'smc'):
        raise ValueError(
            'inference_algorithm must be either is or smc. currently = {}'
            .format(inference_algorithm))

    batch_size = next(iter(observations[0].values())).size(0) \
        if isinstance(observations[0], dict) else observations[0].size(0)

    if return_original_latents or return_latents:
        original_latents = []
    if inference_algorithm == 'smc':
        ancestral_indices = []
    log_weights = []
    log_weights_true = []
    if type(proposal).__name__ == 'Proposal_cnf':
        latent, proposal_log_prob = proposal.sample(observations=observations, time=0, batch_size=batch_size, num_particles=num_particles)
    else:
        proposal_dist = proposal(time=0, observations=observations)
        latent = state.sample(proposal_dist, batch_size, num_particles)
        proposal_log_prob = state.log_prob(proposal_dist, latent)
    latents_bar = [latent]
    initial_log_prob = state.log_prob(initial(), latent)
    emission_log_prob = state.log_prob(
        emission(latents=latents_bar, time=0),
        state.expand_observation(observations[0], num_particles))

    emission_true = copy.deepcopy(emission)
    emission_true.mult=torch.nn.Parameter(torch.tensor(1.0).to(device))
    emission_log_prob_true = state.log_prob(
        emission_true(latents=latents_bar, time=0),
        state.expand_observation(observations[0], num_particles))

    if return_original_latents or return_latents:
        original_latents.append(latent)

    log_weights_true.append(initial_log_prob + emission_log_prob_true -
                       proposal_log_prob)
    log_weights.append(initial_log_prob + emission_log_prob -
                       proposal_log_prob)
    for time in range(1, len(observations)):
        if inference_algorithm == 'smc':
            if args.resampler_type == 'ot' or args.resampler_type == 'soft':
                applied_resampler = resampler(args)
                particle_weights = math.normalize_log_probs(log_weights[-1])+1e-8
                previous_latents_bar = [
                    applied_resampler(torch.unsqueeze(latent, -1), particle_weights)[0].squeeze()
                    for latent in latents_bar[-1:]]
            else:
                ancestral_indices.append(sample_ancestral_index(log_weights[-1]))
                previous_latents_bar = [
                    state.resample(latent, ancestral_indices[-1])
                    for latent in latents_bar[-1:]]
        else:
            previous_latents_bar = latents_bar
        if type(proposal).__name__ == 'Proposal_cnf':
            latent, proposal_log_prob = proposal.sample(previous_latents=previous_latents_bar,
                                                        observations=observations,
                                                        time=time,
                                                        batch_size=batch_size,
                                                        num_particles=num_particles)
        else:
            proposal_dist = proposal(previous_latents=previous_latents_bar,
                                     time=time, observations=observations)
            latent = state.sample(proposal_dist, batch_size, num_particles)
            proposal_log_prob = state.log_prob(proposal_dist, latent)
        latents_bar += [latent]
        transition_log_prob = state.log_prob(
            transition(previous_latents=previous_latents_bar, time=time,
                       previous_observations=observations[:time]),
            latent)
        emission_log_prob = state.log_prob(
            emission(latents=latents_bar, time=time,
                     previous_observations=observations[:time]),
            state.expand_observation(observations[time], num_particles))

        emission_log_prob_true = state.log_prob(
            emission_true(latents=latents_bar, time=time,
                     previous_observations=observations[:time]),
            state.expand_observation(observations[time], num_particles))

        if return_original_latents or return_latents:
            original_latents.append(latent)

        log_weights.append(transition_log_prob + emission_log_prob -
                           proposal_log_prob)
        log_weights_true.append(transition_log_prob + emission_log_prob_true -
                           proposal_log_prob)

    if inference_algorithm == 'smc':
        if return_log_marginal_likelihood:
            temp = torch.logsumexp(torch.stack(log_weights, dim=0), dim=2) - \
                np.log(num_particles)
            log_marginal_likelihood = torch.sum(temp, dim=0)

            temp_true = torch.logsumexp(torch.stack(log_weights_true, dim=0), dim=2) - \
                np.log(num_particles)
            log_marginal_likelihood_true = torch.sum(temp_true, dim=0)
        else:
            log_marginal_likelihood = None

        if return_latents:
            latents = get_resampled_latents(original_latents,
                                                 ancestral_indices)
        else:
            latents = None

        if not return_original_latents:
            original_latents = None

        if return_log_weight:
            log_weight = log_weights[-1]
        else:
            log_weight = None

        if not return_log_weights:
            log_weights = None

        if not return_ancestral_indices:
            ancestral_indices = None
    else:
        if return_log_marginal_likelihood:
            log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
            log_marginal_likelihood = torch.logsumexp(log_weight, dim=1) - \
                np.log(num_particles)
        else:
            log_marginal_likelihood = None

        if return_latents:
            latents = original_latents
        else:
            latents = None

        if return_original_latents:
            original_latents = original_latents

        if return_log_weight:
            if not return_log_marginal_likelihood:
                # already calculated above
                log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
        else:
            log_weight = None

        if not return_log_weights:
            log_weights = None

        ancestral_indices = None
        if return_ancestral_indices:
            raise RuntimeWarning('return_ancestral_indices shouldn\'t be True\
            for is')

    diff = torch.sum(torch.stack(original_latents,dim=0)*(math.normalize_log_probs(torch.stack(log_weights,dim=0))+1e-8),keepdim=True, dim=-1)-\
           torch.stack(true_latents,dim=0)
    loss_rmse = torch.mean(torch.sum(torch.sqrt(torch.sum((diff)**2, dim=-1)),dim=0), dim=0)
    return {'log_marginal_likelihood': log_marginal_likelihood,
            'latents': latents,
            'original_latents': original_latents,
            'log_weight': log_weight,
            'log_weights': log_weights,
            'ancestral_indices': ancestral_indices,
            'last_latent': latent,
            'loss_rmse':loss_rmse}


def get_resampled_latents(latents, ancestral_indices):
    """Resample list of latents.

    Args:
        latents: list of tensors [batch_size, num_particles] or dicts thereof
        ancestral_indices: list where each element is a LongTensor
            [batch_size, num_particles] of length (len(latents) - 1); can
            be empty.

    Returns: list of elements of the same type as latents
    """

    assert(len(ancestral_indices) == len(latents) - 1)
    if isinstance(latents[0], dict):
        temp_value = next(iter(latents[0].values()))
    else:
        temp_value = latents[0]
    batch_size, num_particles = temp_value.size()[:2]

    if temp_value.is_cuda:
        resampled_ancestral_index = torch.arange(0, num_particles).long().\
            cuda().unsqueeze(0).expand(batch_size, num_particles)
    else:
        resampled_ancestral_index = torch.arange(0, num_particles).long().\
            unsqueeze(0).expand(batch_size, num_particles)

    result = []
    for idx, latent in reversed(list(enumerate(latents))):
        result.insert(0, state.resample(latent, resampled_ancestral_index))
        if idx != 0:
            resampled_ancestral_index = torch.gather(
                ancestral_indices[idx - 1],
                dim=1,
                index=resampled_ancestral_index)

    return result


def sample_ancestral_index(log_weight):
    """Sample ancestral index using systematic resampling.

    Args:
        log_weight: log of unnormalized weights, tensor
            [batch_size, num_particles]
    Returns:
        zero-indexed ancestral index: LongTensor [batch_size, num_particles]
    """

    if torch.sum(log_weight != log_weight).item() != 0:
        raise FloatingPointError('log_weight contains nan element(s)')

    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = math.exponentiate_and_normalize(
        log_weight.detach().cpu().numpy(), dim=1)

    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # hack to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights, axis=1, keepdims=True)

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    if log_weight.is_cuda:
        return torch.from_numpy(indices).long().cuda()
    else:
        return torch.from_numpy(indices).long()
