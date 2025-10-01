import os
import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
import cv2

DATA_ROOT = "/home/ab575577/projects_fall_2025/sceneDatasets"
OUTPUT_ROOT = "outputs"

def capture_views(scene_dir):
    # Example: /home/.../00800-TEEsavR23oF/
    folder_name = os.path.basename(scene_dir.rstrip("/"))
    scene_id = folder_name.split("-")[1]
    glb_path = os.path.join(scene_dir, f"{scene_id}.basis.glb")
    nav_path = os.path.join(scene_dir, f"{scene_id}.basis.navmesh")
    output_dir = os.path.join(OUTPUT_ROOT, scene_id)
    os.makedirs(output_dir, exist_ok=True)

    # Habitat config
    cfg = get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = glb_path
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = 640
    cfg.SIMULATOR.SEED = 42
    cfg.freeze()

    with habitat.Env(config=cfg) as env:
        sim = env.sim

        if not sim.pathfinder.load_nav_mesh(nav_path):
            print(f"[WARN] Failed to load navmesh: {nav_path}")
            return

        # Agent 1 position
        state1 = habitat.AgentState()
        state1.position = np.array([1.5, 0.0, 1.5])  # adjust as needed
        state1.rotation = habitat.utils.geometry.quat_from_angle_axis(0.0, np.array([0, 1.0, 0]))
        if not sim.pathfinder.is_navigable(state1.position):
            print(f"[WARN] Agent1 position not navigable in {scene_id}")
            return
        sim.agents[0].set_state(state1)
        obs1 = sim.get_sensor_observations()
        cv2.imwrite(os.path.join(output_dir, "agent1.png"), cv2.cvtColor(obs1["rgb"], cv2.COLOR_RGB2BGR))

        # Agent 2 position
        state2 = habitat.AgentState()
        state2.position = np.array([2.0, 0.0, 1.0])
        state2.rotation = habitat.utils.geometry.quat_from_angle_axis(np.pi/2, np.array([0, 1.0, 0]))
        if not sim.pathfinder.is_navigable(state2.position):
            print(f"[WARN] Agent2 position not navigable in {scene_id}")
            return
        sim.agents[0].set_state(state2)
        obs2 = sim.get_sensor_observations()
        cv2.imwrite(os.path.join(output_dir, "agent2.png"), cv2.cvtColor(obs2["rgb"], cv2.COLOR_RGB2BGR))

        print(f"[INFO] Saved agent1/agent2 views for scene {scene_id}")

if __name__ == "__main__":
    scene_folders = sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    for scene_dir in scene_folders[:10]:  # limit to first 10 scenes for now
        capture_views(scene_dir)
