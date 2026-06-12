"""Render example frames of the Cluttered and FourRooms envs for the poster.

Usage: python make_env_images.py [--seed 42] [--tile_size 96]
Writes PNGs to poster/figures/env_<name>{,_highlight}.png
"""
import argparse
import os

import numpy as np
from PIL import Image

from envs.gym_multigrid.multigrid_envs.cluttered import ClutteredMultiGridFixed15x15
from envs.gym_multigrid.multigrid_envs.fourrooms import FourRoomsEnv15x15


def save_frames(name, env, seed, tile_size, out_dir):
    env.seed(seed)
    env.reset()
    for highlight in (False, True):
        img = env.render(mode='rgb_array', highlight=highlight,
                         tile_size=tile_size)
        suffix = '_highlight' if highlight else ''
        path = os.path.join(out_dir, f'env_{name}{suffix}.png')
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)
        print(f'wrote {path}  ({img.shape[1]}x{img.shape[0]})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tile_size', type=int, default=96)
    parser.add_argument('--out_dir', default='poster/figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_frames('cluttered', ClutteredMultiGridFixed15x15(), args.seed,
                args.tile_size, args.out_dir)
    save_frames('fourrooms', FourRoomsEnv15x15(), args.seed,
                args.tile_size, args.out_dir)


if __name__ == '__main__':
    main()
