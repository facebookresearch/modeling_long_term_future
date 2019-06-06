
## Learning Dynamics Model (CarRacing)

Based on code base for the https://github.com/ikostrikov/pytorch-trpo

Follow similar installations as in https://github.com/ikostrikov/pytorch-trpo.


Requirements:

## Installation

Requirements:
- Python 3.6
- OpenAI Gym
- NumPy
- PyQT5
- PyTorch 0.4.1

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.

Then, clone this repository and install the other dependencies with `pip3`. 

## Generate expert trajectories

We use expert from [World Models](https://worldmodels.github.io/) paper. To generate expert trajectories, run the following commands:

```
cwd=$(pwd)
cd ~
git clone https://github.com/apsdehal-archives/WorldModelsExperiments/
cd WorldModelsExperiments
git checkout expert
cd carracing
python generation_script.py
# This will generate a folder for expert trajectories as `expert_rollouts`. See more details in the repo's README.
cd $cwd
cp ~/WorldModelsExperiments/carracing/expert_rollouts .
```

## Training the student to imitate the expert

To run our model:

    python zforcing_main_carracing.py --batch-size 20 --l2-weight 1.0  --bwd-weight 0.0 --lr 1e-3 --aux-weight-start 0.0005 --aux-weight-end 0.0005 --kld-step 0.0005 --bwd-l2-weight 1. --kld-weight-start 0.2 --clip 1.0

To run our model without auxiliary loss:

    python zforcing_main_carracing.py --batch-size 20 --l2-weight 1.0  --bwd-weight 0.0 --lr 1e-3 --aux-weight-start 0. --aux-weight-end 0. --kld-step 0.0005 --bwd-l2-weight 1. --kld-weight-start 0.2 --clip 1.0

To run recurent decoder baseline:

    python zforcing_main_carracing.py --batch-size 20  --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.000 --bwd-l2-weight 0. --kld-weight-start 0. --l2_weight 1.0 --aux-weight-end 0. --bwd-l2-weight .0

To run recurrent policy baseline:

    python zforcing_main_carracing.py --batch-size 20  --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.000 --bwd-l2-weight 0. --kld-weight-start 0.  --aux-weight-end 0. --bwd-l2-weight .0


## License

Find license in [LICENSE](LICENSE) file.
