# Multi node

## Usage of multi node sampling (DistributedEpiSampler)

Example commands
1. Launching redis server (NOTE: `redis.conf`'s bind should be `0.0.0.0` to be able to access from another server).
2. `python ./run_ppo_distributed_sampler.py --sampler_world_size 4 --redis_host hostname --redis_port port_num`
3. In another terminal `python -m machina.samplers.distributed_epi_sampler --world_size 4 --rank rank` for all rank (NOTE: This command should be executed at `machina/example` dir, because `simple_net.py` is used in this process).


## Usage of multi node training & multi node sampling (ray)

Example commands
1. Set up ray cluster
    - On master node
        - `ray start --head --redis-port <port> --node-ip-address <node_address>`
    - On other nodes
        - `ray start --redis-address <node_address:port>`
2. `python run_ppo_distributed.py --trainer 2 --num_sample_workers 20 --ray_redis_address <node_address:port>`
