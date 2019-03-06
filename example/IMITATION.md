# Imitation Learning
* Behavioral Cloning
* Generative Adversarial Imitation Learning
  
  Original paper: https://arxiv.org/abs/1606.03476
* Adversarial Inverse Reinforcement Learning
  
  Original paper: https://arxiv.org/abs/1710.11248

## Step 1: Place expert epis
You can choose 2 ways
### Download Expert Epis
Download data of expert epis from [here](https://drive.google.com/open?id=1X0c5aC2tylGkwNsZOJxzd88bzAxvK2JG) into `../data/expert_epis`.

For exmaple, run the following code for downloading expert epis  of Pendulum.
```
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1oZ6buPdpFxsp33HY3BVaS2pBHY-1p7OT" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1oZ6buPdpFxsp33HY3BVaS2pBHY-1p7OT" -o data/expert_epis/Pendulum-v0_100epis.pkl
```

### Download Expert Pols
Download data of expert pols from [here](https://drive.google.com/open?id=181I8jwlfRtK5yx2M95c7zZisrEfwfgLw) into `../data/expert_pols`.

Or run RL scripts for learning pol and place pickle file of learned pol into `../data/expert_pols`.

Then, run the following script for making expert epis.
```
python make_expert_epis.py --env_name Pendulum-v0 --pol_fname Pendulum-v0_pol_max.pkl
```
## Step 2: Run script of imitation.
run the following scripts
```
python run_behavior_clone.py
```
```
python run_gail.py
```
```
python run_airl.py
```
See help(`-h`) in order to confirm options.
