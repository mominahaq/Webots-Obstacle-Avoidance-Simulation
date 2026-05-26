cat > ~/robot_nav_repo/README.md << 'EOF'
# 🤖 Webots Robot Navigation with PPO

> Reinforcement learning robot navigation using PPO (Stable Baselines3) in Webots. Custom Gymnasium env with proximity sensors, shaped rewards, and continuous action space.

An e-puck robot learns to navigate to a goal while avoiding obstacles inside a Webots simulation, trained end-to-end with Proximal Policy Optimization.

---

## 📸 Demo

> _Add a GIF or screenshot of your robot navigating here_

---

## 🧠 How It Works

The robot receives a 13-dimensional observation at each timestep and outputs continuous left/right wheel velocities. A shaped reward function guides it toward the goal while penalizing collisions, spinning, and getting stuck.

### Observation Space (13-dim)

| Index | Value |
|-------|-------|
| 0 | `dx` — x delta to goal |
| 1 | `dy` — y delta to goal |
| 2 | Euclidean distance to goal |
| 3 | `cos(angle_diff)` — heading alignment |
| 4 | `sin(angle_diff)` — heading alignment |
| 5–12 | 8 proximity sensor readings (normalized) |

### Action Space (2-dim continuous)

| Index | Value |
|-------|-------|
| 0 | Left wheel speed `[−1, 1]` → scaled to `±6.28 rad/s` |
| 1 | Right wheel speed `[−1, 1]` → scaled to `±6.28 rad/s` |

### Reward Function

| Signal | Amount |
|--------|--------|
| Progress toward goal | `Δdist × 30` |
| Heading alignment | `cos(angle_diff) × 2.5` |
| Collision — danger zone (`>150`) | `−15` |
| Collision — warning zone (`80–150`) | `−0 to −5` (graduated) |
| Anti-spin penalty | `−spin_rate × 0.8` |
| Step penalty | `−0.1` per step |
| Out of bounds | `−50` + teleport back |
| Stuck (`< 0.02 m` in 200 steps) | `−10` |
| Closest-approach bonus | `Δmin_dist × 15` |
| Proximity funnel (`< 0.30 m`) | `(0.30 − dist) × 10` |
| **Goal reached** | **`+500` + up to `+200` efficiency bonus** |

### Environment Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 16 ms |
| Max steps per episode | 10,000 |
| Goal threshold | 0.20 m |
| Arena limit | ±0.42 m |
| Obstacles | 4 circular, radius 0.20 m |

---

---

## ⚙️ Setup

### Prerequisites

- [Webots R2023b+](https://cyberbotics.com/)
- Python 3.10+
- CUDA-capable GPU (optional but recommended)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Webots Python path

The scripts automatically add the default Webots controller path:

/usr/local/webots/lib/controller/python

If your Webots is installed elsewhere, update this line in `train.py` and `run_inference.py`:
```python
sys.path.append('/your/custom/webots/path/lib/controller/python')
```

### 4. Add your world file

Copy your `.wbt` file into the `worlds/` folder and open it in Webots before running any script.

---

## 🚀 Usage

### Train

```bash
python scripts/train.py
```

Training runs for 500,000 timesteps by default. The model is saved to `models/robot_nav_ppo_seed1.zip`.

### Monitor with TensorBoard

```bash
tensorboard --logdir ppo_logs/
```

### Run Inference

```bash
python scripts/run_inference.py
```

Loads the saved model and runs the robot deterministically in the simulation.

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| `stable-baselines3` | ≥ 2.0.0 |
| `gymnasium` | ≥ 0.29.0 |
| `numpy` | ≥ 1.24.0 |
| `torch` | ≥ 2.0.0 |
| `tensorboard` | ≥ 2.13.0 |

---

## 📝 Notes

- **World file** — the `.wbt` file is not included in the repo. Open your Webots project and copy it into `worlds/`.
- **Model files** — `.zip` checkpoints are gitignored by default. Use [Git LFS](https://git-lfs.com/) to track them if you want to share trained weights.
- **Out-of-bounds handling** — the episode does **not** terminate on boundary violations; the robot is teleported back so it has more time to find the goal.
- **Physics lock** — the robot's Z position is locked at `0.0` each step to prevent physics drift in the 2D arena.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
EOF












