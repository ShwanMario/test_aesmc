import aesmc.train as train
import aesmc.losses as losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import unittest
from aesmc.arguments import parse_args
from varname import nameof

def main(args):
    Models = TestModels()
    # Models.test_gaussian(args)
    Models.test_lgssm(args)

def save_data(data_list, name_list, algorithm, lr, num_exps, num_iterations):
    for i, data in enumerate(data_list):
        np.save('./test/test_autoencoder_plots/{}_{}_{}_{}_{}.npy'.format(algorithm, name_list[i], lr, num_exps, num_iterations) ,data)

class TestModels(unittest.TestCase):
    def test_gaussian(self,args):
        device = args.device
        from models import gaussian

        prior_std = 1

        true_prior_mean = 0
        true_obs_std = 1

        prior_mean_init = 2
        obs_std_init = 0.5

        q_init_mult, q_init_bias, q_init_std = 2, 2, 2
        q_true_mult, q_true_bias, q_true_std = gaussian.get_proposal_params(
            true_prior_mean, prior_std, true_obs_std)

        true_prior = gaussian.Prior(true_prior_mean, prior_std).to(device)
        true_likelihood = gaussian.Likelihood(true_obs_std).to(device)

        num_particles = 2
        batch_size = 10
        num_iterations = 2000

        training_stats = gaussian.TrainingStats(logging_interval=500)

        print('\nTraining the \"gaussian\" autoencoder.')
        prior = gaussian.Prior(prior_mean_init, prior_std).to(device)
        likelihood = gaussian.Likelihood(obs_std_init).to(device)
        inference_network = gaussian.InferenceNetwork(
            q_init_mult, q_init_bias, q_init_std).to(device)
        train.train(dataloader=train.get_synthetic_dataloader(
                        true_prior, None, true_likelihood, 1, batch_size),
                    num_particles=num_particles,
                    algorithm='iwae',
                    initial=prior,
                    transition=None,
                    emission=likelihood,
                    proposal=inference_network,
                    num_epochs=1,
                    num_iterations_per_epoch=num_iterations,
                    optimizer_algorithm=torch.optim.SGD,
                    optimizer_kwargs={'lr': 0.01},
                    callback=training_stats, args = args)

        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
        fig.set_size_inches(10, 8)

        mean = training_stats.prior_mean_history
        obs = training_stats.obs_std_history
        mult = training_stats.q_mult_history
        bias = training_stats.q_bias_history
        std = training_stats.q_std_history
        data = [mean] + [obs] + [mult] + [bias] + [std]
        true = [true_prior_mean, true_obs_std, q_true_mult, q_true_bias,
                q_true_std]

        for ax, data_, true_, ylabel in zip(
            axs, data, true, ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']
        ):
            ax.plot(training_stats.iteration_idx_history, data_)
            ax.axhline(true_, color='black')
            ax.set_ylabel(ylabel)
            #  self.assertAlmostEqual(data[-1], true, delta=1e-1)

        axs[-1].set_xlabel('Iteration')
        fig.tight_layout()

        filename = './test/test_autoencoder_plots/gaussian.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

    def test_lgssm(self,args):
        device = args.device
        from models import lgssm
        print('\nTraining the \"linear Gaussian state space model\"'
              ' autoencoder.')
        dim = 2

        initial_loc = torch.zeros([dim,dim]).to(device)
        initial_scale = torch.eye(dim).to(device)
        true_transition_mult = torch.eye(dim).to(device)
        init_transition_mult = (torch.eye(dim)*0.1).to(device)
        transition_scale =  torch.eye(dim).to(device)
        true_emission_mult = (0.5*torch.eye(dim)).to(device)
        init_emission_mult = torch.eye(dim).to(device)

        init_proposal_scale_0 = (0.1*torch.eye(dim)).to(device)
        init_proposal_scale_t = (0.1*torch.eye(dim)).to(device)

        emission_scale = torch.tensor([[0.1**0.5, 0.0], [0.0, 0.1**0.5]]).to(device)
        num_timesteps = 51
        num_test_obs = 10
        test_inference_num_particles = 100
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        num_iterations = 500
        num_particles = 100
        num_experiments = 10
        lr = 0.02
        # http://tuananhle.co.uk/notes/optimal-proposal-lgssm.html
        Gamma_0 = true_emission_mult * initial_scale ** 2 / (emission_scale ** 2 + initial_scale ** 2 * true_emission_mult ** 2)
        optimal_proposal_scale_0 = torch.sqrt(initial_scale**2 - initial_scale**2 * true_emission_mult * Gamma_0)

        Gamma_t = true_emission_mult * transition_scale**2 / (emission_scale**2 + transition_scale**2 * true_emission_mult**2)
        optimal_proposal_scale_t = torch.sqrt(transition_scale**2 - transition_scale**2 * true_emission_mult * Gamma_t)

        algorithms = ['cnf-dpf', 'aesmc', 'bootstrap']
        colors = {'aesmc': 'red', 'cnf-dpf': 'blue', 'bootstrap':'green'}
        dataloader = train.get_synthetic_dataloader(
            lgssm.Initial(initial_loc, initial_scale).to(device),
            lgssm.Transition(true_transition_mult, transition_scale).to(device),
            lgssm.Emission(true_emission_mult, emission_scale).to(device),
            num_timesteps, batch_size)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(15, 10)

        for algorithm in algorithms:
            parameter_error_recorder, posterior_error_recorder, elbo_recorder, ESS_recorder = [], [], [], []
            for i in range(num_experiments):
                training_stats = lgssm.TrainingStats(
                    initial_loc, initial_scale, true_transition_mult,
                    transition_scale, true_emission_mult, emission_scale,
                    num_timesteps, num_test_obs, test_inference_num_particles,
                    saving_interval, logging_interval,algorithm=algorithm, args = args, num_iterations=num_iterations)
                Initial_dist = lgssm.Initial(initial_loc, initial_scale).to(device)
                Transition_dist = lgssm.Transition(init_transition_mult,
                                                        transition_scale).to(device)
                Emission_dist = lgssm.Emission(init_emission_mult,
                                                    emission_scale).to(device)
                if algorithm == 'aesmc':
                    proposal = lgssm.Proposal(init_proposal_scale_0, init_proposal_scale_t, device).to(device)
                elif algorithm == 'cnf-dpf':
                    proposal = lgssm.Proposal_cnf(initial=Initial_dist,
                                                  transition=Transition_dist,
                                                  scale_0=init_proposal_scale_0,
                                                  scale_t=init_proposal_scale_t,
                                                  device=device).to(device)
                elif algorithm == 'bootstrap':
                    proposal = lgssm.Proposal_cnf(initial=Initial_dist,
                                                  transition=Transition_dist,
                                                  scale_0=init_proposal_scale_0,
                                                  scale_t=init_proposal_scale_t,
                                                  device=device,
                                                  type='bootstrap').to(device)
                else:
                    raise ValueError('Please select an algorithm from {aesmc, cnf-dpf, bootstrap}.')
                train.train(dataloader=dataloader,
                            num_particles=num_particles,
                            algorithm=algorithm,
                            initial=Initial_dist,
                            transition=Transition_dist,
                            emission=Emission_dist,
                            # proposal=lgssm.Proposal(optimal_proposal_scale_0,optimal_proposal_scale_t, device).to(device),
                            proposal = proposal,
                            num_epochs=1,
                            num_iterations_per_epoch=num_iterations,
                            optimizer_algorithm=torch.optim.AdamW,
                            optimizer_kwargs={'lr': lr},
                            callback=training_stats,
                            args=args)
                parameter_error_recorder.append(training_stats.p_l2_history)
                posterior_error_recorder.append(training_stats.q_l2_history)
                elbo_recorder.append(np.array(training_stats.loss_history).mean(-1))
                ESS_recorder.append(1/((np.array(training_stats.normalized_log_weights_history)**2).sum(-1)).mean(-1))
                print('Exp. {}/{}, {}, parameter error:{:.6f}+-{:.6f}, posterrior_error:{:.3f}+-{:.3f}'
                      .format(i+1, num_experiments, algorithm,
                              np.array(parameter_error_recorder).mean(0)[-1],
                              np.array(parameter_error_recorder).std(0)[-1],
                              np.array(posterior_error_recorder).mean(0)[-1],
                              np.array(posterior_error_recorder).std(0)[-1],) )

            parameter_error_recorder = np.array(parameter_error_recorder)
            posterior_error_recorder = np.array(posterior_error_recorder)
            elbo_recorder = np.array(elbo_recorder)
            ESS_recorder = np.array(ESS_recorder)

            training_stats.iteration_idx_history = np.array(training_stats.iteration_idx_history)+1
            axs[0,0].plot(training_stats.iteration_idx_history,
                        parameter_error_recorder.mean(0),
                        label=algorithm,
                        color=colors[algorithm])
            axs[0,0].fill_between(training_stats.iteration_idx_history,
                                parameter_error_recorder.mean(0) - parameter_error_recorder.std(0),
                                parameter_error_recorder.mean(0) + parameter_error_recorder.std(0),
                                color=colors[algorithm],
                                alpha=0.3)
            axs[0,1].plot(training_stats.iteration_idx_history,
                        posterior_error_recorder.mean(0),
                        label=algorithm,
                        color=colors[algorithm])
            axs[0,1].fill_between(training_stats.iteration_idx_history,
                                posterior_error_recorder.mean(0) - posterior_error_recorder.std(0),
                                posterior_error_recorder.mean(0) + posterior_error_recorder.std(0),
                                color=colors[algorithm],
                                alpha=0.3)
            axs[1,0].plot(training_stats.iteration_idx_history,
                        elbo_recorder.mean(0),
                        label=algorithm,
                        color=colors[algorithm])
            axs[1,0].fill_between(training_stats.iteration_idx_history,
                                elbo_recorder.mean(0) - elbo_recorder.std(0),
                                elbo_recorder.mean(0) + elbo_recorder.std(0),
                                color=colors[algorithm],
                                alpha=0.3)
            axs[1,1].plot(np.arange(num_timesteps),
                        ESS_recorder[-1].mean(0),
                        label=algorithm,
                        color=colors[algorithm])
            axs[1,1].fill_between(np.arange(num_timesteps),
                                ESS_recorder[:,-1].mean(0) - ESS_recorder[-1].std(0),
                                ESS_recorder[:,-1].mean(0) + ESS_recorder[-1].std(0),
                                color=colors[algorithm],
                                alpha=0.3)
            data_list = [parameter_error_recorder,
                         posterior_error_recorder,
                         elbo_recorder,
                         ESS_recorder]
            data_name_list = ['parameter_error_recorder',
                              'posterior_error_recorder',
                              'elbo_recorder',
                              'ESS_recorder']
            save_data(data_list, data_name_list, algorithm, lr, num_experiments, num_iterations)
        axs[0,0].set_ylabel('$||\\theta - \\theta_{true}||$')
        axs[0, 0].set_xticks([1]+[i for i in np.arange(saving_interval, num_iterations+1, saving_interval)])
        axs[0,1].set_ylabel('Avg. L2 of\nmarginal posterior means')
        axs[0, 1].set_xticks([1] + [i for i in np.arange(saving_interval, num_iterations + 1, saving_interval)])
        axs[1,0].set_ylabel('ELBO')
        axs[1, 0].set_xticks([1] + [i for i in np.arange(saving_interval, num_iterations + 1, saving_interval)])
        axs[1,1].set_ylabel('ESS')
        axs[0,0].set_xlabel('Iteration')
        axs[0,1].set_xlabel('Iteration')
        axs[1,0].set_xlabel('Iteration')
        axs[1,1].set_xlabel('Step')
        axs[0,0].legend()

        for ax in axs:
            for x in ax:
                x.grid(alpha=0.5)
        fig.tight_layout()
        filename = './test/test_autoencoder_plots/lgssm_elbo.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

        self.assertTrue(True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
