import math
import numpy as np
import genesis as gs
import torch
from rsl_rl.env import VecEnv
from genesis.utils.geom import quat_to_xyz, xyz_to_quat, quat_to_R, transform_by_quat, inv_quat, transform_quat_by_quat
import torch.nn.functional as F



def generate_random_positions_and_orientations(base_pos, x_range, y_range, n_envs, device):
    # 生成n_envs个随机x坐标
    random_x = torch.rand(n_envs, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    # 生成n_envs个随机y坐标
    random_y = torch.rand(n_envs, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    # z坐标保持不变
    z_pos = torch.full((n_envs,), base_pos[2], device=device)
    cube_positions = torch.stack([random_x, random_y, z_pos], dim=1)

    # 生成n_envs个随机yaw角度
    random_yaws = torch.rand(n_envs, device=device) * 2 * math.pi
    quaternions = torch.zeros((n_envs, 4), device=device)
    quaternions[:, 0] = torch.cos(random_yaws * 0.5)
    quaternions[:, 3] = torch.sin(random_yaws * 0.5)

    return cube_positions, quaternions


############################继承VecEnv适配RSL RL环境的必须参数#######################################################
# num_envs: int
# num_actions: int
# max_episode_length: int | torch.Tensor
# episode_length_buf: torch.Tensor
# device: torch.device
# cfg: dict | object
# def get_observations(self) -> tuple[torch.Tensor, dict]:
# def reset(self) -> tuple[torch.Tensor, dict]:
# def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
############################继承VecEnv适配RSL RL环境的必须参数#######################################################

class RealmanPickAndPlaceEnv(VecEnv):
    # Realman RM65B机械臂关节角度增量限制（弧度制）
    # 6个手臂关节 + 2个夹爪关节 = 8个自由度
    Q_MIN = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.01, -0.01])
    Q_MAX = torch.tensor([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.01,  0.01])

    # Realman RM65B关节角度范围限制（弧度）
    JOINT_ANGLE_MIN = torch.tensor([-3.1, -2.27, -2.36, -3.1, -2.23, -6.28, 0.0,  0.0])
    JOINT_ANGLE_MAX = torch.tensor([ 3.1,  2.27,  2.36,  3.1,  2.23,  6.28, 0.085, 0.085])
    # 夹爪开合范围限制（米）
    GRIPPER_MIN = 0.0
    GRIPPER_MAX = 0.085

    def __init__(self, cfg: dict | object, num_envs=1, visible=False):
        self.device = gs.device

        # RSL RL 环境的必须参数
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_actions = 8  # 6个手臂关节 + 2个夹爪关节
        self.num_obs = 29  # 3+3+1+3+1+8+4+4+2=29
        self.dt = 0.01  # control frequency on real robot is 100hz
        self.episode_length_s = 1  # 1 seconds
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                box_box_detection=True,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=visible,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/realman/rm65b.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),  # block
                pos=(0.55, 0.0, 0.02),
            ),
            visualize_contact=True,
        )

        # 将关节限制张量移动到 self.device 设备上
        self.Q_MIN = self.Q_MIN.to(self.device)
        self.Q_MAX = self.Q_MAX.to(self.device)
        self.JOINT_ANGLE_MIN = self.JOINT_ANGLE_MIN.to(self.device)
        self.JOINT_ANGLE_MAX = self.JOINT_ANGLE_MAX.to(self.device)

        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))
        self.envs_idx = np.arange(self.num_envs)
        self._initialize_robot_state()

        # 获取current_quat和current_pos
        self.robot_current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_current_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.cube_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.place_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_pos_test = torch.tensor([0.4, 0.0, 0.4], device=self.device).repeat(self.num_envs, 1)
        self.place_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.episode_num = 0

        self.tube_pos = torch.tensor([0.55, 0.0, 0.02], device=self.device).repeat(self.num_envs, 1)
        self.tube_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

    def _initialize_robot_state(self):
        # 初始化放置位置和方向
        self.place_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)

        self.robot_all_dof = torch.arange(8).to(self.device)
        self.motors_dof = torch.arange(6).to(self.device)
        self.fingers_dof = torch.arange(6, 8).to(self.device)
        # Realman RM65B 初始关节角度（弧度），使末端执行器位于工作空间前方
        robot_pos = torch.tensor([0.0, 0.3, -0.8, 0.0, 0.5, 0.0, 0.04, 0.04]).to(self.device)
        robot_pos = robot_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.robot.set_qpos(robot_pos, envs_idx=self.envs_idx)
        self.scene.step()

    # 重置指定单个环境的机器人位置和姿态
    def reset_selected_environments(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # 重置对应环境的机器人位置和姿态（8个关节）
        robot_joint_pos = torch.tensor(
            [0.0, 0.3, -0.8, 0.0, 0.5, 0.0, 0.04, 0.04],
            dtype=torch.float32,
        ).to(self.device)
        self.dof_pos[envs_idx] = robot_joint_pos

        # 设置前6个关节的位置
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx, :6],
            dofs_idx_local=self.motors_dof,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx, 6:8],
            zero_velocity=True,
            dofs_idx_local=self.fingers_dof,
            envs_idx=envs_idx,
        )

        # 使用封装好的函数生成随机位置和方向
        base_pos = torch.tensor([0.55, 0.0, 0.02], device=self.device)
        x_range = torch.tensor([0.54 - self.episode_num, 0.56 + self.episode_num], device=self.device)
        y_range = torch.tensor([-0.01 - self.episode_num, 0.01 + self.episode_num], device=self.device)
        cube_pos, quaternions = generate_random_positions_and_orientations(
            base_pos, x_range, y_range, len(envs_idx), self.device
        )
        self.cube_pos[envs_idx] = cube_pos
        self.cube_quat[envs_idx] = quaternions
        self.cube.set_pos(self.cube_pos[envs_idx], envs_idx=envs_idx)
        self.cube.set_quat(self.cube_quat[envs_idx], envs_idx=envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.episode_num += 0.0001
        if self.episode_num > 0.2:
            self.episode_num = 0
            print('reset')

    def reset(self) -> tuple[torch.Tensor, dict]:
        # 重置所有环境
        self.reset_buf[:] = True
        self.reset_selected_environments(torch.arange(self.num_envs, device=gs.device))
        state, info = self.get_observations()
        self.scene.step()
        return state, info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """执行一步控制循环
        输入: actions [B,8] = 6个手臂关节增量 + 2个夹爪增量
        返回: 状态, 奖励, 终止标志, 信息
        """

        # 1. 限制动作范围actions~[-1,1]
        actions = torch.tanh(actions)

        actions[:, 6:8] = actions[:, 6:8] / 100.0
        delta_actions = torch.clamp(actions, min=self.Q_MIN, max=self.Q_MAX)
        # 2. 获取当前关节位置并计算控制动作
        current_actions = self.robot.get_dofs_position(self.robot_all_dof, self.envs_idx).clone().detach()
        control_actions = current_actions + delta_actions

        # 3. 限制控制动作范围
        control_actions = torch.clamp(control_actions, min=self.JOINT_ANGLE_MIN, max=self.JOINT_ANGLE_MAX)
        self.robot.control_dofs_position(control_actions, self.robot_all_dof, self.envs_idx)
        # 推进物理仿真一步
        self.scene.step()

        # 4. 获取观测和奖励
        states, info = self.get_observations()
        rewards = self.compute_rewards(states, info)
        dones = self._check_termination_conditions(states, info)

        # 获取current_quat和current_pos
        self.robot_current_pos = info["observations"]["gripper_position"]
        self.robot_current_quat = info["observations"]["gripper_quaternion"]

        # 自动重置终止的环境
        self.reset_selected_environments(self.reset_buf.nonzero(as_tuple=False).flatten())
        return states, rewards, dones, info

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        block_position = self.cube.get_pos()
        block_quaternion = self.cube.get_quat()
        gripper_position = self.robot.get_link("Link6").get_pos()
        gripper_quaternion = self.robot.get_link("Link6").get_quat()
        all_dof_pos = self.robot.get_dofs_position(self.robot_all_dof, self.envs_idx)

        # 计算距离时保持维度一致性
        gripper2block_distance = torch.norm(block_position - gripper_position, dim=1, keepdim=True)
        block2target_distance = torch.norm(block_position - self.place_pos_test, dim=1, keepdim=True)

        is_contact_cube = self.get_robot_cube_contacts()

        # 获取左右夹爪到目标的距离
        left_position = self.robot.get_link("Left_Finger").get_pos()
        right_position = self.robot.get_link("Right_Finger").get_pos()
        left_distance = torch.norm(left_position - block_position, dim=1, keepdim=True)
        right_distance = torch.norm(right_position - block_position, dim=1, keepdim=True)

        # 计算绝对值差
        distance_diff = torch.abs(left_distance - right_distance)

        states_dict = {
            "observations": {
                "block_quaternion": block_quaternion,
                "block_position": block_position,
                "gripper_position": gripper_position,
                "gripper_quaternion": gripper_quaternion,
                "robot_all_dof": all_dof_pos,
                "gripper2block_distance": gripper2block_distance,
                "block2target_distance": block2target_distance,
                "place_pos": self.place_pos_test,
                "is_contact_cube": is_contact_cube,
                "distance_diff": distance_diff,
            }
        }

        states = torch.cat([
            block_position,           # 3
            gripper_position,         # 3
            gripper2block_distance,   # 1
            self.place_pos_test,      # 3
            block2target_distance,    # 1
            all_dof_pos,              # 8
            gripper_quaternion,       # 4
            block_quaternion,         # 4
            is_contact_cube,          # 2
        ], dim=1)

        return states, states_dict

    # 定义奖励函数，计算每个环境的奖励pick and place
    def compute_rewards(self, states, states_dict):
        # --- 奖励最大值与衰减参数 ---
        R_approach_max = 1.0
        decay_approach = 5.0

        R_pre_grasp_max = 100.0
        R_lift_max = 200.0
        R_move_max = 1000.0
        decay_move = 3.0

        R_done_max = 20000.0
        # --- 阈值与时间惩罚 ---
        contact_thresh = 0.15
        lift_thresh = 0.005
        target_thresh = 0.1
        time_weight = 1e-3

        obs = states_dict["observations"]
        g2b = obs["gripper2block_distance"].squeeze(-1)
        b2t = obs["block2target_distance"].squeeze(-1)
        z_pos = obs["block_position"][:, 2]
        distance_diff = obs["distance_diff"].squeeze(-1)

        # --- 实时接触检测 ---
        contact_bool = self.get_robot_cube_contacts().sum(dim=1)

        # --- 阶段信号 ---
        # Phase 1: always available
        approach_reward = R_approach_max * torch.tanh(decay_approach * g2b)

        # 两者距离差值越大则惩罚越大
        alignment_reward = -R_approach_max * torch.tanh(decay_approach * distance_diff)

        # Phase 2: pre-grasp bonus
        pre_grasp_signal = (g2b < contact_thresh) & (contact_bool > 1)
        reward_pre_grasp = R_pre_grasp_max * ((g2b < contact_thresh).float() + contact_bool) / 3

        # Phase 3: grasp validation & lift
        lift_signal = ((z_pos - self.tube_pos[0, 2]) > lift_thresh) & pre_grasp_signal
        reward_lift = lift_signal.float() * R_lift_max
        reward_move = R_move_max * torch.tanh(decay_move * b2t) * lift_signal

        # Phase 4: placement completion
        done_signal = (b2t < target_thresh).float() * pre_grasp_signal
        reward_done = R_done_max * done_signal

        # --- 时间惩罚 ---
        time_penalty = -time_weight * self.episode_length_buf
        # --- 总奖励 ---
        total_reward = (
            approach_reward
            + alignment_reward
            + reward_pre_grasp
            + reward_lift
            + reward_move
            + reward_done
            + time_penalty
        )

        return total_reward / (R_approach_max + R_pre_grasp_max + R_lift_max + R_move_max + R_done_max)

    def get_robot_cube_contacts(self):
        LEFT_LINK = 7
        RIGHT_LINK = 8

        def _check_contact(link_id):
            contacts = self.robot.get_contacts(self.cube)

            valid_mask = contacts['valid_mask']
            link_a = contacts["link_a"]

            link_tensor = torch.tensor([link_id], device=link_a.device).repeat(self.num_envs, 1)
            isin_a = torch.logical_and(torch.isin(link_a, link_tensor), valid_mask)
            return isin_a.any(dim=1).float()

        contact_left = _check_contact(LEFT_LINK)
        contact_right = _check_contact(RIGHT_LINK)
        return torch.stack([contact_left, contact_right], dim=1)

    def get_robot_contacts(self, object):
        contacts = self.robot.get_contacts(object)
        valid_mask = contacts['valid_mask']
        link_b = contacts["link_b"]
        link_a = contacts["link_a"]

        if object == self.cube:
            link = torch.tensor([8], device=link_b.device).repeat(self.num_envs, 1)
        elif object == self.plane:
            link = torch.tensor([0], device=link_b.device).repeat(self.num_envs, 1)

        isin_a = torch.logical_and(torch.isin(link_a, link), valid_mask)
        isin_b = torch.logical_and(torch.isin(link_b, link), valid_mask)

        is_contact = (isin_a | isin_b).float().sum(dim=1)
        contact_bool = (is_contact > 0).float()
        return contact_bool

    def get_orientation_reward(self, cube_quat, gripper_quat):
        # 获取旋转矩阵
        cube_quat_R = quat_to_R(cube_quat)
        gripper_quat_R = quat_to_R(gripper_quat)

        # 获取Z轴向量并归一化
        cube_z_axis = F.normalize(cube_quat_R[:, :, 2], dim=-1)
        gripper_z_axis = F.normalize(gripper_quat_R[:, :, 2], dim=-1)

        # 安全计算点积并限制范围
        dot_product = torch.sum(cube_z_axis * gripper_z_axis, dim=-1).clamp(-1.0, 1.0)
        # 反向平行，即点积负数越小奖励越大
        return torch.exp(-dot_product)

    # 定义终止条件，判断每个环境是否达到终止状态
    def _check_termination_conditions(self, states, states_dict):
        self.episode_length_buf += 1
        time_exceeded = self.episode_length_buf > self.max_episode_length
        obs = states_dict["observations"]
        block_to_target = obs["block2target_distance"].squeeze(-1)
        task_complete = block_to_target < 0.2
        contact_bool = self.get_robot_contacts(self.plane)
        # 转为bool类型
        contact_bool = contact_bool.bool()
        self.reset_buf = time_exceeded | task_complete | contact_bool

        return self.reset_buf.clone()


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32", logging_level='warning')

    env_cfg = {
        "num_actions": 8,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 2.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
    }

    env = RealmanPickAndPlaceEnv(cfg=env_cfg, num_envs=4, visible=True)
    state, info = env.reset()
    for i in range(1000):
        # 随机生成动作变化
        actions = 2 * torch.rand((env.num_envs, 8), device=env.device) - 1
        states, rewards, dones, info = env.step(actions)
