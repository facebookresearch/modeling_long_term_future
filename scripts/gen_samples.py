#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
import pickle
import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--room", type=int, default=8,
                    help="size of the room")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")

def write_samples(all_samples_obs, all_samples_actions, filename):
    # write to pickle file
    import ipdb; ipdb.set_trace()
    all_data = list(zip(all_samples_obs, all_samples_actions))
    output = open(filename, "wb")
    pickle.dump(all_data, output)
    output.close()
    return True

def evaluate(agent, env, episodes):
    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}	
    expert_actions = [] 
    expert_obs = []
    all_samples_obs = []
    all_samples_actions = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        expert_obs = []
        expert_actions = []
        expert_obs.append(obs['image'])
        num_frames = 0
        returnn = 0

        while not(done):
            action = agent.get_action(obs)
            expert_actions.append(action)
            obs, reward, done, _ = env.step(action)
            
            expert_obs.append(obs['image'])
            agent.analyze_feedback(reward, done)
            
            num_frames += 1
            returnn += reward
        
        import ipdb; ipdb.set_trace()
        all_samples_obs.append(expert_obs)
        all_samples_actions.append(expert_actions)
        if (ep + 1) % 1000 == 0:
            print ("iteration: " + str(ep))
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)

    filename = 'data/' + args.env + 'start_flag_room_'+ str(args.room)+ '_' + str(episodes) + '_samples.dat' 
    write_samples(all_samples_obs, all_samples_actions, filename)
    return logs


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
    if args.seed is None:
        args.seed = 0 if args.model is not None else 1

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Generate environment

    env = gym.make(args.env)
    env.seed(args.seed)

    # Define agent

    agent = utils.load_agent(args, env)

    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos

        args.episodes = len(agent.demos)

    # Run the agent

    start_time = time.time()
    
    logs = evaluate(agent, env, args.episodes)

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    n = 10
    print("{} worst episodes:".format(n))
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
