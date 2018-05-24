import numpy as np
import torch
from suggest_method.expert_data.base import BaseData
from machina.utils import np2torch, torch2torch
import os




class ExpertData(BaseData):
    def __init__(self, expert_path, test_size):
        self.expert_data = np.load(expert_path)
        self.obs = np2torch(self.expert_data['obs'])
        self.acs = np2torch(self.expert_data['acs'])
        self.num_of_traj = self.expert_data['obs'].shape[0]
        self.num_of_step = self.expert_data['obs'].shape[1]
        full_indices = np.array([[traj_indice, step_indice] for traj_indice in range(self.num_of_traj) for step_indice in range(self.num_of_step-1)], dtype=np.int)
        full_indices_shuffled = np.random.permutation(full_indices)
        self.indices_train, indices_test = np.array_split(full_indices_shuffled, [int((1-test_size)*len(full_indices_shuffled))])
        indices_test_traj = np2torch(indices_test.T[0]).long()
        indices_test_step = np2torch(indices_test.T[1]).long()
        self.test_data = dict(
                        obs=self.obs[indices_test_traj,indices_test_step],
                        next_obs=self.obs[indices_test_traj, indices_test_step + 1],
                        acs=self.acs[indices_test_traj, indices_test_step]
                )

    def iterate(self, batch_size):
        indices_split = np.array_split(self.indices_train, len(self.indices_train) // batch_size)
        for iteration, indices_minibatch in enumerate(indices_split):
            indices_minibatch_traj = np2torch(indices_minibatch.T[0]).long()
            indices_minibatch_step = np2torch(indices_minibatch.T[1]).long()
            yield iteration,\
                    dict(
                        obs=self.obs[indices_minibatch_traj,indices_minibatch_step],
                        next_obs=self.obs[indices_minibatch_traj, indices_minibatch_step + 1],
                        acs=self.acs[indices_minibatch_traj, indices_minibatch_step]
                )


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='expert_data_npz')
    parser.add_argument('--file_name', type=str, default='halfcheetah_rllab_3232_noBN_0202pol_max_HalfCheetah-v1_30trajs.npz')
    args = parser.parse_args()
    expertdata = ExpertData(os.path.join(os.getcwd(),args.dir, args.file_name), 0.1)
    for iteration, batch in expertdata.iterate(batch_size=100):
#    for batch in expertdata.iterate(batch_size=3,num_of_step=1, epoch=2):
        print('iteration={}, batchsize={}'.format(iteration, len(batch['obs'])))
    print(expertdata.obs)
    print(expertdata.obs.shape)
