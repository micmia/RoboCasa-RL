import os
import sys

robocasa_path = os.path.join(os.path.dirname(__file__), 'robocasa')
if robocasa_path not in sys.path:
    sys.path.insert(0, robocasa_path)

import robosuite
import robocasa
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

from env.custom_pnp_counter_to_cab import register_custom_env
register_custom_env()

def main():
    print("Creating 'MyPnPCounterToCab' environment...")
    
    camera_names = [
        "robot0_agentview_center",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
        "robot0_frontview",
        "robot0_robotview"
    ]
    
    env = robosuite.make(
        env_name="MyPnPCounterToCab",
        robots="PandaOmron",
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,               # Needs to be True to get camera frames
        camera_names=camera_names,
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    
    print("Environment created. Resetting...")
    obs = env.reset()
    
    # ── Verify Which Cameras Are Actually Output ──
    # If a camera doesn't exist, it might not be in the obs dict.
    available_cams = [cam for cam in camera_names if f"{cam}_image" in obs]
    print(f"Requested {len(camera_names)} cameras. Retrieved {len(available_cams)} valid camera streams.")
    
    # Prepare Matplotlib figure
    n = len(available_cams)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    if n == 0:
        print("No camera observations found! Check camera_names.")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    displays = []
    
    # Initialize the plots with the first observation
    for ax, cam in zip(axes[:n], available_cams):
        ax.set_title(cam.replace("robot0_", "").replace("_", " ").title(), fontsize=10)
        ax.axis("off")
        
        img = obs[f"{cam}_image"]
        
        # Mujoco returns shape (H, W, 1) for Depth or (H, W, 3) for RGB
        # If the image does not have 3 channels, duplicate to 3 or plot as grayscale
        if img.shape[-1] != 3:
            img = np.stack([img.squeeze()] * 3, axis=-1)
            
        # Mujoco renders upside down, so flip it
        img = img[::-1]
        
        displays.append(ax.imshow(img))

    # Turn off unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    print(f"Starting random actions. Action dimension: {env.action_dim}")

    step_counter = [0]
    def update(frame):
        """ Matplotlib animation update callback """
        if step_counter[0] >= 1000:
            plt.close(fig)
            return displays
            
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)
        env.render()
        
        step_counter[0] += 1
        
        if done:
            print("Episode finished early. Resetting...")
            obs = env.reset()
            
        for d, cam in zip(displays, available_cams):
            if f"{cam}_image" in obs:
                img = obs[f"{cam}_image"]
                if img.shape[-1] != 3:
                    img = np.stack([img.squeeze()] * 3, axis=-1)
                img = img[::-1]
                d.set_array(img)
                
        return displays

    # Start matplotlib loop
    anim = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Done visualizing.")

if __name__ == "__main__":
    main()
