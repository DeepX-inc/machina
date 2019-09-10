# Multi node

## Multi-node sampling

### DistributedEpiSampler
- Example commands
1. Launching redis server (NOTE: `redis.conf`'s bind should be `0.0.0.0` to be able to access from another server).
2. `python ./run_ppo_distributed_sampler.py --sampler_world_size 4 --redis_host hostname --redis_port port_num`
3. In another terminal `python -m machina.samplers.distributed_epi_sampler --world_size 4 --rank rank` for all rank (NOTE: This command should be executed at `machina/example` dir, because `simple_net.py` is used in this process).

### RaySampler
1. Set up ray cluster
    - On master node
        - `ray start --head --redis-port <port> --node-ip-address <node_address>`
    - On other nodes
        - `ray start --redis-address <node_address:port>`
2. `python run_ppo_distributed_ray.py --num_parallel 20 --ray_redis_address <node_address:port>`

## Multi-node (multi-GPUs) training

### torch.distributed
Example commands
1. `python -m torch.distributed.launch --nproc_per_node 2 --master_addr 192.168.10.4 --master_port 12341 --nnode 1 --node_rank 0 ./run_ppo_distributed.py `

### ray + torch.distributed
Example commands
1. Set up ray cluster
    - On master node
        - `ray start --head --redis-port <port> --node-ip-address <node_address>`
    - On other nodes
        - `ray start --redis-address <node_address:port>`
2. `python run_ppo_distributed_ray.py --trainer 2 --num_parallel 20 --ray_redis_address <node_address:port>`
