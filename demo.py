import os
import time
import glob
import io
import base64
import numpy as np
import gymnasium as gym
import highway_env
import cv2
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
# from IPython import get_ipython
from IPython.display import HTML, display

try:
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False
    
PROJECT_DIR = os.path.join(os.getcwd(), "Project")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
VIDEO_DIR = os.path.join(PROJECT_DIR, "videos")

for folder in [PROJECT_DIR, LOG_DIR, MODEL_DIR, VIDEO_DIR]:
    os.makedirs(folder, exist_ok=True)

def show_video(video_path_pattern):
    ''' 
    Display the latest video in the NoteBook
    '''
    mp4_list = glob.glob(video_path_pattern)
    if not mp4_list:
        print(f" No video found for: {video_path_pattern}")
        return
    video_path = mp4_list[0]
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded_video = base64.b64encode(video_data).decode("ascii")
    if IN_JUPYTER:
        display(HTML(data=f'''
            <video width="640" height="480" controls>
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
            </video>
        '''))
    else:
        print(f" Video saved: {video_path}")

def make_env(env_name="highway-fast-v0"):
    '''
    Created a Configure for the Enviornment using Gym for traning
    '''
    env = gym.make(env_name, render_mode="rgb_array")
    if hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure({
            "lanes_count": 4,
            "vehicles_count": 20,
            "duration": 40,
            "collision_reward": -2,
            "high_speed_reward": 1,
            "reward_speed_range": [20, 30],
            "action": {"type": "DiscreteMetaAction"}
        })
    return Monitor(env)

def record_with_opencv(frames, out_file="demo.avi", fps=15):
    '''
    OpennCv package has been used to record the video in 3 different format - MP4,AVC1,XVID
    '''
    if not frames:
        print("No frames to record.")
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f"Created video in these location: {out_file}")

if __name__ == "__main__":
    ENVIRONMENTS = ["highway-fast-v0"]
    LEARNING_RATES = [0.0005, 0.001]
    GAMMAS = [0.95, 0.8]

    TOTAL_TIMESTEPS = 150000  # 原项目中是150000，这里为了节省时间可以适当减少来测试效果

    results = {}

    for env_name in ENVIRONMENTS:
        for lr in LEARNING_RATES:
            for gm in GAMMAS:
                run_id = f"{env_name}_DQN_lr{lr}_gm{gm}"
                print(f"TRAINING RUN: {run_id}")

                # Create environment
                vec_env = make_env(env_name)

                tb_log_dir = os.path.join(LOG_DIR, run_id)

                # Create DQN with chosen LR, gamma
                model = DQN(
                    "MlpPolicy",
                    vec_env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=lr,
                    gamma=gm,
                    verbose=0,
                    tensorboard_log=tb_log_dir,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    exploration_fraction=0.7,
                    device="auto",  # "auto" will use GPU if available, otherwise CPU. Can be set to "cuda" or "cpu".
                )

                # Train
                model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

                # Evaluate
                mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5)

                if mean_reward > 0:
                    success_rate = (mean_reward - std_reward) / mean_reward
                else:
                    success_rate = 0

                print(f" Completed run for env={env_name}, gamma={gm}, lr={lr}")
                print(f"   Final average reward: {mean_reward:.2f}")
                print(f"   Success rate: {success_rate:.2f} \n")

                # Save model
                model_path = os.path.join(MODEL_DIR, f"{run_id}_model")
                model.save(model_path)
                print(f"[INFO] Model saved at: {model_path}\n")

                # Record short video
                demo_env = make_env(env_name)
                if hasattr(demo_env.unwrapped, "config"):
                    demo_env.unwrapped.config["simulation_frequency"] = 15

                frames = []
                obs, info = demo_env.reset()
                done = truncated = False
                for step in range(300):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = demo_env.step(action)
                    frames.append(demo_env.render())
                    if done or truncated:
                        obs, info = demo_env.reset()
                demo_env.close()

                video_file = os.path.join(VIDEO_DIR, f"{run_id}_{int(time.time())}.avi")
                record_with_opencv(frames, out_file=video_file, fps=15)
                show_video(video_file)

                results[run_id] = {
                    "env": env_name,
                    "lr": lr,
                    "gamma": gm,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "success_rate": success_rate
                }

    print("\nFinal Result of All model trained Above\n")
    for run_id, result_data in results.items():
        print(f"--- {run_id} ---")
        print(f"  Mean Reward: {result_data['mean_reward']:.2f} +/- {result_data['std_reward']:.2f}")
        print(f"  Success Rate: {result_data['success_rate']:.2f}\n")