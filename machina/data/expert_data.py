import numpy as np
import torch
from suggest_method.expert_data.base import BaseData
from machina.utils import np2torch, torch2torch
import os




class ExpertData(BaseData):
    def __init__(self, expert_path):
        self.expert_data = np.load(expert_path)
        self.obs = np2torch(self.expert_data['obs'])
        self.acs = np2torch(self.expert_data['acs'])
        self.num_of_traj = self.expert_data['obs'].shape[0]
        self.num_of_step = self.expert_data['obs'].shape[1]

    def iterate(self, batch_size, epoch=10):
        full_indices = np.arange(self.num_of_traj * self.num_of_step, dtype=np.int) + 1
        full_indices = full_indices[np.where(full_indices%self.num_of_step!=0)]
        full_indices = full_indices -1
        full_indices_shuffled = np.random.permutation(full_indices)
        indices_minibatches = np.array_split(full_indices_shuffled, len(full_indices) // batch_size)
        for iteration, indices_minibatch in enumerate(indices_minibatches):
            indices_minibatch_traj = np2torch(indices_minibatch // self.num_of_step).long()
            indices_minibatch_step = np2torch(indices_minibatch % self.num_of_step).long()
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
    expertdata = ExpertData(os.path.join(os.getcwd(),args.dir, args.file_name))
    for iteration, batch in expertdata.iterate(batch_size=100,epoch=2):
#    for batch in expertdata.iterate(batch_size=3,num_of_step=1, epoch=2):
        print('iteration={}, batchsize={}'.format(iteration, len(batch['obs'])))
    print(expertdata.obs)
    print(expertdata.obs.shape)
