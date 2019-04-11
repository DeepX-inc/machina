# Multi node

## Usage of multi node sampling

Example commands
1. Launching redis server (NOTE: `redis.conf`'s bind should be `0.0.0.0` to be able to access from another server).
2. `python ./run_ppo_multi_node.py --sampler_world_size 4 --redis_host hostname --redis_port port_num`
3. In another terminal `python -m machina.samplers.distributed_epi_sampler --world_size 4 --rank rank` for all rank (NOTE: This command should be executed at `machina/example` dir, because `simple_net.py` is used in this process).
