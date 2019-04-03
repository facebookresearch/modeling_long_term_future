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

    git clone https://github.com/nke001/iclr_babyai
    cd babyai
    pip3 install --editable .

Create a new conda env using env.yml in the repo

## Training teacher

We use the Half Cheetah task.

First train the teacher using TRPO on HalfCheetah-v2.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH/.mujoco/mjpro150/bin
    python main.py --env-name "HalfCheetah-v2"


## Generate expert trajectories

Our students is trained on high-dimensional images, we need to ask expert to render images.

Rendering Mujoco headlessly on the server

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

    python zforcing_main_cheetah.py --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.0001 --bwd-l2-weight 1. --kld-weight-start 0.2  --aux-weight-end 0.0001 --bwd-l2-weight 1.0

To run the baseline

    python zforcing_main_cheetah.py --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.000 --bwd-l2-weight 0. --kld-weight-start 0.  --aux-weight-end 0. --bwd-l2-weight .0


## License

Find license in [LICENSE](LICENSE) file.
