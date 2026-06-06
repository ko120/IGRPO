import argparse
import os
import random
import torch
import numpy as np
import wandb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import utils
from multiagent_metacontroller import MultiAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
        help='Name of environment.')
    parser.add_argument(
        '--mode', type=str, default='ppo',
        help="Name of experiment. Can be 'ppo'")
    parser.add_argument(
        '--debug', action=argparse.BooleanOptionalAction,
        help="If used will disable wandb logging.")
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed.")
    parser.add_argument(
        '--keep_training', action=argparse.BooleanOptionalAction,
        help="If used will continue training from previous checkpoint.")
    parser.add_argument(
        '--visualize', action=argparse.BooleanOptionalAction,
        help="If used will run visualization only.")
    parser.add_argument(
        '--video_dir', type=str, default='videos',
        help="Name of location to store videos.")
    parser.add_argument(
        '--load_checkpoint_from', type=str, default=None,
        help="Path to find model checkpoints to load")
    parser.add_argument(
        '--save_path', type=str, default=None,
        help="Directory to save model checkpoints. Defaults to Ray's ~/ray_results/.")
    parser.add_argument(
        '--wandb_project', type=str, default='multigrid-ippo',
        help="Name of wandb project.")
    parser.add_argument(
        '--algorithm', type=str, default=None, choices=['IPPO', 'MAPPO'],
        help="Algorithm: IPPO (local obs) or MAPPO (shared direction info). Overrides config.")
    parser.add_argument(
        '--backward', action="store_true")
    parser.add_argument(
        '--global_norm', action=argparse.BooleanOptionalAction, default=None,
        help="GRPO: normalize rewards globally across all agents and timesteps (--global_norm) or per-timestep (--no-global_norm).")
    parser.add_argument(
        '--use_return', action=argparse.BooleanOptionalAction, default=None,
        help="GRPO: compute discounted return-to-go as advantage (--use_return) or use normalized reward directly (--no-use_return, default).")
    parser.add_argument(
        '--n_episodes', type=int, default=None,
        help="Override total training episodes (e.g. small value for quick Colab runs).")
    return parser.parse_args()


def get_metacontroller_class(config):
    return MultiAgent


def initialize(mode, env_name, debug, visualize, seed, wandb_project):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = utils.generate_parameters(
        mode=mode, domain=env_name, debug=(debug or visualize),
        seed=seed, wandb_project=wandb_project)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = utils.make_env(config)

    metacontroller_class = get_metacontroller_class(config)

    return device, config, env, metacontroller_class


def main(args):
    device, config, env, metacontroller_class = initialize(
        args.mode, args.env_name, args.debug, args.visualize,
        args.seed, args.wandb_project)

    if args.algorithm:
        config.algorithm = args.algorithm
    if args.save_path:
        # Resolve save_path relative to project root (avoids root-level permission errors)
        config.save_path = os.path.join(PROJECT_ROOT, args.save_path.lstrip('/'))
    if args.backward:
        config.backward = args.backward
    if args.global_norm is not None:
        config.global_norm = args.global_norm
    if args.use_return is not None:
        config.use_return = args.use_return
    if args.n_episodes is not None:
        config.n_episodes = args.n_episodes

    # Ensure if you're logging to wandb, it's to the right project
    if not args.debug and not args.visualize:
        if not args.wandb_project:
            print('ERROR: when logging to wandb, must specify a valid wandb project.')
            exit(1)

    if args.visualize:
        agent = metacontroller_class(
            config, env, device, training=False)
        agent.load_models(model_path=args.load_checkpoint_from)
        agent.visualize(env, args.mode, args.video_dir)
        print('A video of the trained policies being tested in the environment '
              'has been generated.')
        exit(0)

    # Train Model
    agent = metacontroller_class(
        config, env, device, debug=args.debug)

    if args.keep_training and args.load_checkpoint_from:
        agent.load_models(model_path=args.load_checkpoint_from)

    agent.train(env)


if __name__ == '__main__':
    main(parse_args())
