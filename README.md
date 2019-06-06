This repo contains code for our paper [Learning Dynamics Model in Reinforcement Learning by Incorporating the Long Term Future](https://arxiv.org/abs/1903.01599)

The code base contains multiple branches.

- The main branch contains experiments for the BabyAI tasks.
- The mujoco branch contains experiments for the Mujoco tasks.
- The carracing branch contains experiments for CarRacing task.

Based on code base for the BabyAI project at Mila. https://github.com/mila-iqia/babyai

Follow similar installations as in https://github.com/mila-iqia/babyai.


Requirements:
## Installation

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- PyQT5
- PyTorch 0.4.1+

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.

Then, clone this repository and install the other dependencies with `pip3`:

    git clone https://github.com/facebookresearch/modeling_long_term_future.git
    cd modeling_long_term_future
    pip3 install --editable .

Create a new conda env using env.yml in the repo

## Training teacher

We use the BabyAI Pickup-Unlock game.

First train the teacher (for imitation learning) using PPO with curriculum learning. Start with a room size of 6 and then work our way up to room size of 15.


    python3 -m scripts.train_curclm. --env BabyAI-UnlockPickup-v0  --algo ppo   --arch cnn1 --tb --seed 1 --save-interval 10 --room-size 6
    python3 -m scripts.train_curclm. --env BabyAI-UnlockPickup-v0  --algo ppo   --arch cnn1 --tb --seed 1 --save-interval 10 --room-size 8 --model MODEL_ROOM6_PRETRAINED
    python3 -m scripts.train_curclm. --env BabyAI-UnlockPickup-v0  --algo ppo   --arch cnn1 --tb --seed 1 --save-interval 10 --room-size 10 --model MODEL_ROOM8_PRETRAINED
    python3 -m scripts.train_curclm. --env BabyAI-UnlockPickup-v0  --algo ppo   --arch cnn1 --tb --seed 1 --save-interval 10 --room-size 12 --model MODEL_ROOM10_PRETRAINED
    python3 -m scripts.train_curclm. --env BabyAI-UnlockPickup-v0  --algo ppo   --arch cnn1 --tb --seed 1 --save-interval 10 --room-size 15 --model MODEL_ROOM12_PRETRAINED


## Generate expert trajectories

Generate expert trajectories from the experts trained using curriculum learning

    mnkdir data
    python3 -m scripts.gen_samples --episodes 10000 --env BabyAI-UnlockPickup-v0 --model pretrained_model_room_10 --room 10


## Training the student to imitate the expert

To run our model

    python3 -m scripts.zforcing_main_state_dec --env BabyAI-UnlockPickup-v0 --datafile EXPERT_DATA_TO_LOAD --model MODEL_NAME --eval-episodes 100 --eval-interval 200  --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.0001 --bwd-l2-weight 1. --kld-weight-start 0.2  --aux-weight-end 0.0001  --room 10


To run the baseline

    python3 -m scripts.zforcing_main_state_dec --datafile EXPERT_DATA_TO_LOAD --env BabyAI-UnlockPickup-v0 --model MODEL_NAME --eval-episodes 100 --eval-interval 200  --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.000 --aux-weight-end 0.0 --room 10


## License

Find license in [LICENSE](LICENSE) file.
