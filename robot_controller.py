import gymnasium as gym
import numpy as np
from gymnasium import spaces
from controller import Supervisor

# ── Constants ──────────────────────────────────────────────────────────────────
TIME_STEP           = 16
MAX_SPEED           = 6.28
MAX_STEPS           = 10000
GOAL_THRESHOLD      = 0.20
ARENA_LIMIT         = 0.42

SENSOR_WARN         = 80.0
SENSOR_DANGER       = 150.0

OBSTACLE_POSITIONS  = [
    (-0.25, -0.25),
    (-0.25,  0.25),
    ( 0.15, -0.25),
    ( 0.1,   0.1 )
]
OBSTACLE_RADIUS     = 0.20
FIXED_Z             = 0.0

STUCK_WINDOW        = 200
STUCK_THRESHOLD     = 0.02


class RobotNavEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.robot      = Supervisor()
        self.timestep   = TIME_STEP

        self.robot_node  = self.robot.getSelf()
        self.target_node = self.robot.getFromDef("TARGET")

        self.sensors = []
        for i in range(8):
            s = self.robot.getDevice(f"ps{i}")
            s.enable(TIME_STEP)
            self.sensors.append(s)

        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.step_count       = 0
        self.prev_distance    = None
        self.min_distance     = None
        self.position_history = []

        self._init_robot_z()

    def _init_robot_z(self):
        rp = self.robot_node.getPosition()
        self.robot_node.getField("translation").setSFVec3f([rp[0], rp[1], FIXED_Z])
        self.robot_node.resetPhysics()

    def _lock_z(self):
        rp = self.robot_node.getPosition()
        if abs(rp[2] - FIXED_Z) > 0.0001:
            self.robot_node.getField("translation").setSFVec3f([rp[0], rp[1], FIXED_Z])

    def _is_safe_position(self, x, y):
        for (ox, oy) in OBSTACLE_POSITIONS:
            if np.sqrt((x - ox)**2 + (y - oy)**2) < OBSTACLE_RADIUS:
                return False
        tp = self.target_node.getPosition()
        if np.sqrt((x - tp[0])**2 + (y - tp[1])**2) < OBSTACLE_RADIUS:
            return False
        if abs(x) > 0.35 or abs(y) > 0.35:
            return False
        return True

    def _get_safe_spawn(self):
        for _ in range(300):
            x = np.random.uniform(-0.30, 0.30)
            y = np.random.uniform(-0.30, 0.30)
            if self._is_safe_position(x, y):
                return x, y
        return 0.0, -0.30

    def _get_robot_heading(self):
        rot = self.robot_node.getField("rotation").getSFRotation()
        ax, ay, az, angle = rot
        if abs(ay) > 0.5:
            return angle * np.sign(ay)
        if abs(az) > 0.5:
            return angle * np.sign(az)
        return 0.0

    def _get_angle_to_goal(self):
        rp = self.robot_node.getPosition()
        tp = self.target_node.getPosition()
        goal_dir = np.arctan2(tp[1] - rp[1], tp[0] - rp[0])
        heading  = self._get_robot_heading()
        diff     = goal_dir - heading
        return float(np.arctan2(np.sin(diff), np.cos(diff)))

    def _get_distance(self):
        rp = self.robot_node.getPosition()
        tp = self.target_node.getPosition()
        return float(np.sqrt((rp[0] - tp[0])**2 + (rp[1] - tp[1])**2))

    def _get_obs(self):
        rp  = self.robot_node.getPosition()
        tp  = self.target_node.getPosition()
        dx  = tp[0] - rp[0]
        dy  = tp[1] - rp[1]
        dist       = np.sqrt(dx**2 + dy**2)
        angle_diff = self._get_angle_to_goal()
        sensors    = [s.getValue() / 1000.0 for s in self.sensors]
        return np.array(
            [dx, dy, dist, np.cos(angle_diff), np.sin(angle_diff)] + sensors,
            dtype=np.float32
        )

    def _is_stuck(self):
        if len(self.position_history) < STUCK_WINDOW:
            return False
        old_pos = self.position_history[-STUCK_WINDOW]
        cur_pos = self.robot_node.getPosition()
        moved   = np.sqrt((cur_pos[0] - old_pos[0])**2 + (cur_pos[1] - old_pos[1])**2)
        return moved < STUCK_THRESHOLD

    def _teleport_back(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.robot_node.resetPhysics()
        for _ in range(10):
            self.robot.step(self.timestep)
        x, y = self._get_safe_spawn()
        self.robot_node.getField("translation").setSFVec3f([x, y, FIXED_Z])
        self.robot_node.getField("rotation").setSFRotation([0, 1, 0, 0])
        self.robot_node.resetPhysics()
        for _ in range(10):
            self.robot.step(self.timestep)
        self._lock_z()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.robot_node.resetPhysics()

        x, y = self._get_safe_spawn()
        self.robot_node.getField("translation").setSFVec3f([x, y, FIXED_Z])
        self.robot_node.getField("rotation").setSFRotation([0, 1, 0, 0])
        self.robot_node.resetPhysics()

        for _ in range(10):
            self.robot.step(self.timestep)
        self._lock_z()

        self.step_count       = 0
        self.prev_distance    = self._get_distance()
        self.min_distance     = self.prev_distance
        self.position_history = []

        return self._get_obs(), {}

    def step(self, action):
        left_speed  = float(action[0]) * MAX_SPEED
        right_speed = float(action[1]) * MAX_SPEED
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        self.robot.step(self.timestep)
        self.step_count += 1

        obs         = self._get_obs()
        distance    = self._get_distance()
        sensor_vals = [s.getValue() for s in self.sensors]
        max_sensor  = max(sensor_vals)
        rp          = self.robot_node.getPosition()
        angle_diff  = self._get_angle_to_goal()

        self.position_history.append((rp[0], rp[1]))

        reward     = 0.0
        terminated = False
        truncated  = False

        # ── 1. Progress reward ─────────────────────────────────────────────
        progress = (self.prev_distance - distance) * 30.0
        reward  += progress

        # ── 2. Heading alignment ───────────────────────────────────────────
        reward += np.cos(angle_diff) * 2.5

        # ── 3. Graduated collision penalty ────────────────────────────────
        if max_sensor > SENSOR_DANGER:
            reward -= 15.0
        elif max_sensor > SENSOR_WARN:
            ratio   = (max_sensor - SENSOR_WARN) / (SENSOR_DANGER - SENSOR_WARN)
            reward -= ratio * 5.0

        # ── 4. Anti-spin penalty ───────────────────────────────────────────
        spin_rate = abs(left_speed - right_speed) / (2.0 * MAX_SPEED)
        reward   -= spin_rate * 0.8

        # ── 5. Step penalty ────────────────────────────────────────────────
        reward -= 0.1

        # ── 6. Out-of-bounds — teleport back and CONTINUE episode ──────────
        # KEY FIX: episode does NOT terminate — agent gets more time to find goal
        if abs(rp[0]) > ARENA_LIMIT or abs(rp[1]) > ARENA_LIMIT:
            reward -= 50.0
            self._teleport_back()
            self.prev_distance = self._get_distance()
            self.min_distance  = self.prev_distance
            print("⚠️ Out of bounds — teleported back!")
            return self._get_obs(), reward, False, False, {}

        # ── 7. Stuck penalty ──────────────────────────────────────────────
        if self._is_stuck():
            reward -= 10.0

        # ── 8. Closest-approach bonus ──────────────────────────────────────
        if distance < self.min_distance:
            reward           += (self.min_distance - distance) * 15.0
            self.min_distance = distance

        # ── 9. Proximity funnel ────────────────────────────────────────────
        if distance < 0.30:
            reward += (0.30 - distance) * 10.0

        # ── 10. Goal reward ────────────────────────────────────────────────
        if distance < GOAL_THRESHOLD:
            efficiency = max(0.0, (MAX_STEPS - self.step_count) / MAX_STEPS)
            bonus      = 500.0 + efficiency * 200.0
            reward    += bonus
            terminated = True
            print(f"✅ Goal reached in {self.step_count} steps! "
                  f"Efficiency bonus: +{efficiency * 200:.1f}")

        # ── 11. Timeout ────────────────────────────────────────────────────
        if self.step_count >= MAX_STEPS:
            truncated = True

        self.prev_distance = distance
        return obs, reward, terminated, truncated, {}
