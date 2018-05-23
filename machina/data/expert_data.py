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

    def iterate_once(self, batch_size):
        indices_traj = np2torch(np.random.choice(np.arange(self.obs.shape[0]), batch_size))
        indices_step = np2torch(np.random.choice(np.arange(self.obs.shape[1]-1)  , batch_size))
        indices_next_step = indices_step + 1
        return dict(
            obs=self.obs[indices_traj, indices_step],
            next_obs=self.obs[indices_traj, indices_next_step],
            acs=self.acs[indices_traj, indices_step]
        )


    def iterate_nstep(self, batch_size, num_of_step, epoch=1):
        for i in range(num_of_step):
            nstep_obs = torch2torch(torch.zeros((num_of_step, batch_size, self.obs.shape[2]))).float()
            nstep_acs = torch2torch(torch.zeros((num_of_step, batch_size, self.acs.shape[2]))).float()
            indices_traj = np2torch(np.random.choice(np.arange(self.obs.shape[0]), batch_size))
            indices_step = np2torch(np.random.choice(np.arange(self.obs.shape[1] - 1), batch_size))
            for i in range(num_of_step):
                nstep_obs[i] = self.obs[indices_traj, indices_step + i]
                nstep_acs[i] = self.acs[indices_traj, indices_step + i]
            return dict(
                nstep_obs=nstep_obs,
                nstep_acs=nstep_acs
            )

    def iterate(self, batch_size, num_of_step, epoch=1):
        if num_of_step==1:
            for _ in range(epoch):
                batch = self.iterate_once(batch_size)
                yield batch
        else:
            for _ in range(epoch):
                batch = self.iterate_nstep(batch_size, num_of_step)
                yield batch


if __name__=='__main__':
    expertdata = ExpertData(os.path.join(os.getcwd(),'halfcheetah_rllab_3232_noBN_0202pol_max_HalfCheetah-v1_30trajs.npz'))
#    for batch in expertdata.iterate(batch_size=3,epoch=2):
#        print(batch['obs'])
#    print(expertdata.obs)
#    print(expertdata.obs.shape)
    for batch in expertdata.iterate_nstep(batch_size=5, num_of_step=1):
        print(batch['nstep_obs'])