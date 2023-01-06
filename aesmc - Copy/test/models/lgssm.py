import copy
import aesmc
import numpy as np
import pykalman
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import sys
sys.path.append('../test/')

class Initial(nn.Module):
    def __init__(self, loc, scale):
        super(Initial, self).__init__()
        self.loc = loc
        self.scale = scale

    def forward(self):
        return torch.distributions.Normal(self.loc, self.scale)


class Transition(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition, self).__init__()
        self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())#init_mult#
        self.scale = scale#nn.Parameter(torch.Tensor([scale]).squeeze())#

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(
                self.mult * previous_latents[-1], self.scale),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Emission(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission, self).__init__()
        self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())#torch.Tensor([init_mult]).squeeze()#
        self.scale = scale

    def forward(self, latents=None, time=None, previous_observations=None):
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(self.mult * latents[-1], self.scale),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Proposal(nn.Module):
    def __init__(self, scale_0, scale_t, device):
        super(Proposal, self).__init__()
        self.scale_0 = nn.Parameter(torch.Tensor([scale_0]).squeeze())#scale_0#
        self.scale_t = nn.Parameter(torch.Tensor([scale_t]).squeeze())#scale_t#
        self.lin_0 = nn.Linear(1, 1, bias=False).to(device)
        self.lin_t = nn.Linear(2, 1, bias=False).to(device)

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0].unsqueeze(-1)).squeeze(-1),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_t(torch.cat(
                        [previous_latents[-1].unsqueeze(-1),
                         observations[time].view(-1, 1, 1).expand(
                            -1, num_particles, 1
                         )],
                        dim=2
                    ).view(-1, 2)).squeeze(-1).view(-1, num_particles),
                    scale=self.scale_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Proposal_cnf(nn.Module):
    def __init__(self, transition, initial, scale_0, scale_t, device, type = 'planar'):
        super(Proposal_cnf, self).__init__()
        self.scale_0 = nn.Parameter(torch.Tensor([scale_0]).squeeze())#scale_0#
        self.scale_t = nn.Parameter(torch.Tensor([scale_t]).squeeze())#scale_t#
        self.transition = transition
        self.initial = initial

        self.lin_0 = nn.Linear(1, 1, bias=False).to(device)
        self.lin_t = nn.Linear(2, 1, bias=False).to(device)

        self.type = type
        if self.type == 'planar':
            self.u = nn.Parameter(torch.tensor(0.1).to(device))#torch.tensor(0.1).to(device)#
            self.b = nn.Parameter(torch.tensor(0.1).to(device))#torch.tensor(0.1).to(device)#
            self.w = nn.Parameter(torch.tensor(0.1).to(device))#torch.tensor(0.1).to(device)#
            self.planar = Planar(dim = 1)
        elif self.type == 'radial':
            self.radial_flow = Radial(dim = 1)
    def sample(self, previous_latents=None, time=None, observations=None, batch_size = 10, num_particles = 100):
        if self.type != 'bootstrap':
            if time == 0:
                dist_0 = aesmc.state.set_batch_shape_mode(
                    torch.distributions.Normal(
                        loc=self.lin_0(observations[0].unsqueeze(-1)).squeeze(-1),
                        scale=self.scale_0),
                    aesmc.state.BatchShapeMode.BATCH_EXPANDED)
                samples = aesmc.state.sample(dist_0, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(dist_0, samples)
                return samples, proposal_log_prob
            else:
                dist_t = aesmc.state.set_batch_shape_mode(
                    torch.distributions.Normal(
                        loc=self.lin_t(torch.cat(
                            [previous_latents[-1].unsqueeze(-1),
                             observations[time].view(-1, 1, 1).expand(
                                -1, num_particles, 1
                             )],
                            dim=2
                        ).view(-1, 2)).squeeze(-1).view(-1, num_particles),
                        scale=self.scale_t),
                    aesmc.state.BatchShapeMode.FULLY_EXPANDED)
                proposal_samples = aesmc.state.sample(dist_t, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(dist_t, proposal_samples)

                if self.type == 'planar':
                    proposal_samples = proposal_samples + self.u*torch.tanh(self.w*proposal_samples +
                                                                            self.b*observations[time].view(-1, 1).expand(-1, num_particles))
                    log_det = (1+self.u*self.w*(1-torch.tanh(self.w*proposal_samples +
                                                             self.b*observations[time].view(-1, 1).expand(-1, num_particles))**2)).abs().log()
                elif self.type == 'radial':
                    proposal_samples, log_det = self.radial_flow(proposal_samples)
                elif self.type == 'normal':
                    log_det = 0.0
                else:
                    raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
                proposal_log_prob = proposal_log_prob - log_det.squeeze(-1)
        elif self.type == 'bootstrap':
            if time == 0:
                initial_samples = aesmc.state.sample(self.initial(), batch_size, num_particles)
                initial_log_prob = aesmc.state.log_prob(self.initial(), initial_samples)
                return initial_samples, initial_log_prob
            else:
                transition_dist = self.transition(previous_latents=previous_latents)
                proposal_samples = aesmc.state.sample(transition_dist, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(
                    self.transition(previous_latents=previous_latents), proposal_samples)
        else:
            raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
        return proposal_samples, proposal_log_prob

class FCNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )


    def forward(self, x):
        return self.network(x.float())

functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

class Planar(nn.Module):
    def __init__(self, dim, nonlinearity=torch.tanh):
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x, observation):
        if self.h in (F.elu, F.leaky_relu):
            u = self.u
        elif self.h == torch.tanh:
            scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
            u = self.u + scal * self.w / torch.norm(self.w) ** 2
        else:
            raise NotImplementedError("Non-linearity is not supported.")
        lin = torch.unsqueeze(x @ self.w, -1) + self.b * observation
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z.squeeze(-1), log_det


class Radial(nn.Module):
    """
    Radial flow.
        z = f(x) = = x + β h(α, r)(z − z0)
    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim):
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(self, dim):
        init.uniform_(self.x0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det

class RealNVP_cond_0(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.obser_dim = obser_dim
        self.t1 = base_network(self.obser_dim, self.dim, hidden_dim)
        self.s1 = base_network(self.obser_dim, self.dim, hidden_dim)
        self.t2 = base_network(self.obser_dim, self.dim, hidden_dim)
        self.s2 = base_network(self.obser_dim, self.dim, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, obser):
        t1_transformed = self.t1(torch.cat([obser],dim=-1))
        s1_transformed = self.s1(torch.cat([obser],dim=-1))
        x = t1_transformed + x * torch.exp(s1_transformed)
        log_det = torch.sum(s1_transformed, dim=-1)
        return x, log_det

    # def inverse(self, z, obser):
    #     lower, upper = z[:,:self.dim_1], z[:,self.dim_1:]
    #     t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
    #     s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
    #     lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
    #     t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
    #     s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
    #     upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
    #     x = torch.cat([lower, upper], dim=1)
    #     log_det = torch.sum(-s1_transformed, dim=1) + \
    #               torch.sum(-s2_transformed, dim=1)
    #     return x, log_det

class RealNVP_cond_t(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.obser_dim = obser_dim
        self.t1 = base_network(self.dim+self.obser_dim, self.dim, hidden_dim)
        self.s1 = base_network(self.dim+self.obser_dim, self.dim, hidden_dim)
        self.t2 = base_network(self.dim+self.obser_dim, self.dim, hidden_dim)
        self.s2 = base_network(self.dim+self.obser_dim, self.dim, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, previous_x, obser):
        t1_transformed = self.t1(torch.cat([previous_x,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([previous_x,obser],dim=-1))
        x = t1_transformed + x * torch.exp(s1_transformed)
        log_det = torch.sum(s1_transformed, dim=-1)
        return x, log_det
    #
    # def inverse(self, z, obser):
    #     lower, upper = z[:,:self.dim_1], z[:,self.dim_1:]
    #     t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
    #     s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
    #     lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
    #     t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
    #     s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
    #     upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
    #     x = torch.cat([lower, upper], dim=1)
    #     log_det = torch.sum(-s1_transformed, dim=1) + \
    #               torch.sum(-s2_transformed, dim=1)
    #     return x, log_det

def lgssm_true_posterior(observations, initial_loc, initial_scale,
                         transition_mult, transition_bias, transition_scale,
                         emission_mult, emission_bias, emission_scale):
    ssm_parameter_lists = [initial_loc, initial_scale,
    transition_mult, transition_scale,
    emission_mult, emission_scale]

    initial_loc, initial_scale, \
    transition_mult, transition_scale, \
    emission_mult, emission_scale = [ssm_parameter.cpu().detach().numpy() for
                                                    ssm_parameter in ssm_parameter_lists if torch.is_tensor(ssm_parameter)]
    kf = pykalman.KalmanFilter(
        initial_state_mean=[initial_loc],
        initial_state_covariance=[[initial_scale**2]],
        transition_matrices=[[transition_mult]],
        transition_offsets=[transition_bias],
        transition_covariance=[[transition_scale**2]],
        observation_matrices=[[emission_mult]],
        observation_offsets=[emission_bias],
        observation_covariance=[[emission_scale**2]])

    return kf.smooth([observation[0].cpu() for observation in observations])


class TrainingStats(object):
    def __init__(self, initial_loc, initial_scale, true_transition_mult,
                 transition_scale, true_emission_mult, emission_scale,
                 num_timesteps, num_test_obs, test_inference_num_particles,
                 saving_interval=100, logging_interval=100, algorithm='is',args=None, num_iterations=500):
        device = args.device
        self.true_transition_mult = true_transition_mult
        self.true_emission_mult = true_emission_mult
        self.test_inference_num_particles = test_inference_num_particles
        self.saving_interval = saving_interval
        self.logging_interval = logging_interval
        self.p_l2_history = []
        self.q_l2_history = []
        self.normalized_log_weights_history = []
        self.iteration_idx_history = []
        self.loss_history = []
        self.initial = Initial(initial_loc, initial_scale).to(device)
        self.true_transition = Transition(true_transition_mult,
                                          transition_scale).to(device)
        self.true_emission = Emission(true_emission_mult, emission_scale).to(device)
        dataloader = aesmc.train.get_synthetic_dataloader(self.initial,
                                                          self.true_transition,
                                                          self.true_emission,
                                                          num_timesteps,
                                                          num_test_obs)
        self.test_obs = next(iter(dataloader))
        self.true_posterior_means = [None] * num_test_obs
        for test_obs_idx in range(num_test_obs):
            latent = [[l[test_obs_idx]] for l in self.test_obs[0]]
            observations = [[o[test_obs_idx]] for o in self.test_obs[1]]
            self.true_posterior_means[test_obs_idx] = np.reshape(
                lgssm_true_posterior(observations, initial_loc, initial_scale,
                                     self.true_transition_mult, 0,
                                     transition_scale, self.true_emission_mult,
                                     0, emission_scale)[0],
                (-1,))
        if algorithm == 'iwae':
            self.algorithm = 'is'
        else:
            self.algorithm = 'smc'

        self.args = args
        self.device = args.device
        self.num_iterations = num_iterations
    def __call__(self, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal):
        if epoch_iteration_idx % self.saving_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            self.p_l2_history.append(np.linalg.norm(
                np.array([transition.mult.item(), emission.mult.item()]) -
                np.array([self.true_transition_mult.cpu().detach().numpy(), self.true_emission_mult.cpu().detach().numpy()])
            ))
            latents = self.test_obs[0]
            latents = [latent.to(self.device).unsqueeze(-1) for latent in latents]
            inference_result = aesmc.inference.infer(
                self.algorithm, self.test_obs[1], self.initial,
                self.true_transition, self.true_emission, proposal,
                self.test_inference_num_particles,args=self.args, true_latents=latents,return_log_marginal_likelihood=True)
            posterior_means = aesmc.statistics.empirical_mean(
                torch.cat([latent.unsqueeze(-1) for
                           latent in [original_latents.cpu() for original_latents in inference_result['original_latents']]], dim=2),
                inference_result['log_weight'].cpu()).detach().numpy()
            self.q_l2_history.append(np.mean(np.linalg.norm(
                self.true_posterior_means - posterior_means, axis=1)))
            normalized_weights = aesmc.math.normalize_log_probs(torch.stack(inference_result['log_weights'],dim=0))+1e-8
            self.normalized_log_weights_history.append(normalized_weights.cpu().detach().numpy())
            self.loss_history.append(inference_result['log_marginal_likelihood'].cpu().detach().numpy())
            self.iteration_idx_history.append(epoch_iteration_idx)

        if epoch_iteration_idx % self.logging_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            print('Iteration {}:'
                  ' Loss = {:.3f},'
                  ' parameter error = {:.6f},'
                  ' posterior mean error = {:.3f}'
                  .format(epoch_iteration_idx,loss,
                          self.p_l2_history[-1],
                          self.q_l2_history[-1]))
