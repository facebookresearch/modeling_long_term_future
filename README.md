

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

Then, clone this repository and install the other dependencies with `pip3`:

    git clone https://github.com/nke001/iclr_mujoco
    cd iclr_mujoco
    conda create -n ENV_NAME env.yml


## Training teacher

We use the Half Cheetah task.

First train the teacher using TRPO on HalfCheetah-v2.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH/.mujoco/mjpro150/bin
    python main.py --env-name "HalfCheetah-v2"

## Generate expert trajectories

Our students is trained on high-dimensional images, we need to ask expert to render images.

Rendering Mujoco headlessly on the server

* Add these to the file
    ```from pyvirtualdisplay import Display
    display_ = Display(visible=0, size=(550, 500))
    display_.start()
    image = env.render(mode="rgb_array")

* Also, in the mujoco/mujoco_env.py, cchange the def render() function to the following
    ```
    def render(self, mode='human'):
        if mode == 'rgb_array':
            data = self.sim.render(500, 500)
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()


The cheetah is moving, so need to have the camera tracking the movements of cheetah. Need to change a few things in mujoco_py


Generate 10k expert trajectories from the experts.

    python gen_samples_cheetah.py --num-samples 100000


## Training the student to imitate the expert

To run our model

    python zforcing_main_cheetah.py --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.0001 --bwd-l2-weight 1. --kld-weight-start 0.2  --aux-weight-end 0.0001 --bwd-l2-weight 1.0

To run the baseline

    python zforcing_main_cheetah.py --bwd-weight 0.0 --lr 1e-4 --aux-weight-start 0.000 --bwd-l2-weight 0. --kld-weight-start 0.  --aux-weight-end 0. --bwd-l2-weight .0


## License

Find license in [LICENSE](LICENSE) file.
