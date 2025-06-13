import argparse
import functools

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

import mediapy


def progress(num_steps, metrics):
  print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", type=str, default="panda", choices=["panda", "leap"])
  parser.add_argument("--num_steps", type=int, default=10_000_000)
  args = parser.parse_args()

  if args.env == "panda":
    env_name = "PandaPickCubeOrientation"
  elif args.env == "leap":
    env_name = "LeapCubeReorient"
  else:
    raise ValueError(f"Unknown env: {args.env}")
  
  env = registry.load(env_name)
  env_cfg = registry.get_default_config(env_name)

  ppo_params = manipulation_params.brax_ppo_config(env_name)
  ppo_training_params = dict(ppo_params)
  ppo_training_params["num_envs"] = 1024
  ppo_training_params["num_timesteps"] = args.num_steps

  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      progress_fn=progress,
      seed=1
  )

  make_inference_fn, params, metrics = train_fn(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )

  print("Training done.")

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  rng = jax.random.PRNGKey(42)
  rollout = []
  n_episodes = 1

  for _ in range(n_episodes):
    state = jit_reset(rng)
    rollout.append(state)
    for i in range(env_cfg.episode_length):
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_step(state, ctrl)
      rollout.append(state)

  render_every = 1
  frames = env.render(rollout[::render_every])
  rewards = [s.reward for s in rollout]

  mediapy.write_video(f"{args.env}.mp4", frames, fps=1.0 / env.dt / render_every)
