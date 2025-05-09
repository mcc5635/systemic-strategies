from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import argparse
import sys

from hummingbot.strategy.hjlp.hjlp_gym_env import HJLPEnv
from hummingbot.strategy.hjlp.hjlp_system import hJLPSystem

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train hJLP RL agent')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Number of timesteps to train')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    parser.add_argument('--plot-only', action='store_true', help='Only plot training metrics from logs dir')
    parser.add_argument('--logs-dir', type=str, default=None, help='Logs directory for plotting')
    return parser.parse_args()

def create_simulation_data():
    """Generate realistic simulation data"""
    def make_orderbook(mid, spread=2, depth=3):
        bids = [(mid - spread/2 - i, 1000 + 500*i) for i in range(depth)]
        asks = [(mid + spread/2 + i, 1000 + 500*i) for i in range(depth)]
        return {
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0],
            "best_ask": asks[0][0],
        }
    return {
        "spot_prices": {
            "SOL": np.random.normal(100, 5),
            "USDC": 1.0,
            "BTC": np.random.normal(65000, 500),
            "ETH": np.random.normal(3200, 50),
        },
        "funding_rates": {
            "BTC": np.random.normal(0.00005, 0.0001),
            "ETH": np.random.normal(0.00007, 0.00012),
            "SOL": np.random.normal(0.0001, 0.0002),
            "SOL-PERP": np.random.normal(0.0001, 0.0002),
        },
        "orderbook": {
            "BTC": make_orderbook(65000),
            "ETH": make_orderbook(3200),
            "SOL": make_orderbook(100),
            "USDC": make_orderbook(1, spread=0.01),
        },
        "jlp_supply": np.random.normal(1_000_000, 100_000),
    }

def plot_training_metrics(log_dir):
    logger.info("Plotting training metrics...")
    csv_path = f"{log_dir}/monitor.csv"
    try:
        data = np.genfromtxt(csv_path, delimiter=',', comments='#', skip_header=1, invalid_raise=False)
        # Remove rows that are not the expected length
        if data.ndim == 1:
            data = data[None, :]  # handle single row
        data = data[data.shape[1] == 3] if data.shape[1] == 3 else data
        if data.shape[0] == 0 or data.shape[1] < 2:
            logger.error(f"No valid data to plot in {csv_path}")
            return
        timesteps = data[:, 1]
        rewards = data[:, 0]
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, rewards)
        plt.title('Training Rewards Over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.savefig(f"{log_dir}/training_rewards.png")
        plt.close()
        logger.info(f"Saved training rewards plot to {log_dir}/training_rewards.png")
    except Exception as e:
        logger.error(f"Could not plot training metrics: {e}")

def main():
    args = parse_args()
    
    # Plot-only mode
    if hasattr(args, 'plot_only') and args.plot_only:
        plot_training_metrics(args.logs_dir)
        sys.exit(0)

    # Create directories for saving models and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sim" if args.simulation else "live"
    models_dir = f"models/hjlp_{mode}_{timestamp}"
    logs_dir = f"logs/hjlp_{mode}_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize environment with your wallet address
    user_address = "4uZcVvQ4ihHfjP7n4KvxfqhoZJLUJj81eV7f6QZaDUTJ"
    logger.info(f"Initializing hJLP system with wallet address: {user_address}")
    logger.info(f"Running in {'simulation' if args.simulation else 'live'} mode")

    try:
        hjlp_system = hJLPSystem(user_address=user_address)
        if args.simulation:
            # Override API methods with simulation data
            hjlp_system.fetch_spot_prices = lambda: create_simulation_data()["spot_prices"]
            hjlp_system.fetch_funding_rates = lambda: create_simulation_data()["funding_rates"]
            hjlp_system.fetch_orderbook_state = lambda: create_simulation_data()["orderbook"]
            hjlp_system.fetch_jlp_supply = lambda: create_simulation_data()["jlp_supply"]
            logger.info("Using simulated market data")

        env = HJLPEnv(hjlp_system)
        logger.info("Successfully initialized hJLP system and environment")

        # Wrap environment with Monitor for logging
        env = Monitor(env, logs_dir)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        logger.info("Environment wrapped with Monitor and VecNormalize")

        # Test environment with random actions
        logger.info("Testing environment with random actions...")
        obs = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
            logger.info(f"Random action {i+1}/10 - Reward: {reward[0]:.2f}")
            if done[0]:
                break
            if not args.simulation:
                time.sleep(args.delay)  # Add delay between API calls only in live mode

        # Set up callbacks
        eval_env = HJLPEnv(hjlp_system)
        eval_env = Monitor(eval_env, logs_dir)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{models_dir}/best_model",
            log_path=f"{logs_dir}/eval_logs",
            eval_freq=1000,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"{models_dir}/checkpoints",
            name_prefix="hjlp_model"
        )

        # Initialize PPO agent with custom parameters
        logger.info("Initializing PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

        # Train the agent
        logger.info(f"Starting training for {args.timesteps} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback]
        )

        # Save the final model and environment normalization
        logger.info("Saving final model and environment normalization...")
        model.save(f"{models_dir}/final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")

        # Test the trained agent
        logger.info("Testing trained agent...")
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1
            logger.info(f"Step {step}, Reward: {reward[0]:.2f}, Total Reward: {total_reward:.2f}")
            if not args.simulation:
                time.sleep(args.delay)  # Add delay between API calls only in live mode

        logger.info(f"Testing complete. Total reward: {total_reward:.2f}")

        # Close environments
        env.close()
        eval_env.close()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hJLP RL agent')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Number of timesteps to train')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    parser.add_argument('--plot-only', action='store_true', help='Only plot training metrics from logs dir')
    parser.add_argument('--logs-dir', type=str, default=None, help='Logs directory for plotting')
    args = parser.parse_args()
    if args.plot_only and args.logs_dir:
        plot_training_metrics(args.logs_dir)
    else:
        main() 