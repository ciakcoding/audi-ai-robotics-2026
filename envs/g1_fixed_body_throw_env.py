
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

class G1FixedBodyThrowEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, xml_path=None, episode_time=1.8, control_dt=0.02, action_scale=0.5, learned_release=False, scripted_release_time=0.65, target_pos=(2.0,-0.25,1.0)):
        super().__init__()
        if xml_path is None: xml_path=Path(__file__).resolve().parents[1]/'assets'/'unitree_g1_throw'/'scene_throw.xml'
        self.xml_path=Path(xml_path)
        self.model=mujoco.MjModel.from_xml_path(str(self.xml_path)); self.data=mujoco.MjData(self.model)
        self.episode_time=float(episode_time); self.control_dt=float(control_dt); self.frame_skip=max(1,int(round(self.control_dt/self.model.opt.timestep)))
        self.action_scale=float(action_scale); self.learned_release=bool(learned_release); self.scripted_release_time=float(scripted_release_time); self.target_pos=np.array(target_pos,dtype=np.float64)
        self.ball_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'throw_ball'); self.target_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'throw_target'); self.hold_eq_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_EQUALITY,'hold_throw_ball'); self.ball_joint_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'throw_ball_free')
        missing=[]
        if self.ball_body_id<0: missing.append('throw_ball')
        if self.target_body_id<0: missing.append('throw_target')
        if self.hold_eq_id<0: missing.append('hold_throw_ball')
        if self.ball_joint_id<0: missing.append('throw_ball_free')
        if missing: raise RuntimeError('Missing from G1 throwing scene: '+', '.join(missing)+'. Run scripts/create_g1_throw_scene.py first.')
        self.hold_body_id=int(self.model.eq_obj1id[self.hold_eq_id]); self.ball_qpos_adr=int(self.model.jnt_qposadr[self.ball_joint_id]); self.ball_qvel_adr=int(self.model.jnt_dofadr[self.ball_joint_id]); self.hold_relpose=self._load_hold_relpose()
        self.arm_joint_names=self._find_right_arm_joint_names()
        if not self.arm_joint_names: raise RuntimeError('Could not find right arm joints. Run scripts/inspect_g1.py and edit envs/g1_fixed_body_throw_env.py.')
        self.arm_joint_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.arm_joint_names]
        self.arm_qpos_adr=np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids]); self.arm_qvel_adr=np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_actuator_ids=self._find_arm_actuator_ids(); self.n_arm=len(self.arm_joint_names)
        self.action_space=spaces.Box(-1,1,shape=(self.n_arm+1,),dtype=np.float32)
        obs_dim=self.n_arm+self.n_arm+3+3+3+(self.n_arm+1)+1+1; self.observation_space=spaces.Box(-np.inf,np.inf,shape=(obs_dim,),dtype=np.float32)
        self.prev_action=np.zeros(self.n_arm+1); self.nominal_qpos=np.zeros(self.model.nq); self.nominal_ctrl=np.zeros(self.model.nu); self._init_nominal_pose()
        self.step_count=0; self.released=False; self.release_time=None; self.best_dist=np.inf
    def _find_right_arm_joint_names(self):
        all_names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(self.model.njnt)]; all_names=[n for n in all_names if n]
        preferred=['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
        if all(n in all_names for n in preferred): return preferred
        candidates=[n for n in all_names if 'right' in n.lower() and any(k in n.lower() for k in ['shoulder','elbow','wrist','arm'])]
        ordered=[]
        for key in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']:
            for n in candidates:
                if key in n.lower() and n not in ordered: ordered.append(n)
        return ordered or candidates
    def _find_arm_actuator_ids(self):
        ids=[]
        for jname in self.arm_joint_names:
            jid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,jname); found=-1
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid,0]==jid: found=aid; break
            if found<0: raise RuntimeError(f'Could not find actuator for joint {jname}. Run scripts/inspect_g1.py.')
            ids.append(found)
        return np.array(ids,dtype=np.int32)
    def _init_nominal_pose(self):
        mujoco.mj_resetData(self.model,self.data)
        if self.model.nkey>0: mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data); self.nominal_qpos[:]=self.data.qpos[:]; self.nominal_ctrl[:]=0
        for aid in range(self.model.nu):
            trnid=self.model.actuator_trnid[aid,0]
            if trnid>=0:
                qadr=self.model.jnt_qposadr[trnid]
                if qadr<self.model.nq: self.nominal_ctrl[aid]=self.data.qpos[qadr]
    def _load_hold_relpose(self):
        root=ET.parse(self.xml_path).getroot(); weld=root.find("./equality/weld[@name='hold_throw_ball']")
        if weld is None: raise RuntimeError('Could not find hold_throw_ball weld in throwing scene XML.')
        relpose=np.fromstring(weld.attrib.get('relpose','0 0 0 1 0 0 0'),sep=' ',dtype=np.float64)
        if relpose.shape!=(7,): raise RuntimeError('hold_throw_ball relpose must have 7 numbers.')
        return relpose
    def _place_ball_in_hand(self):
        hand_pos=self.data.xpos[self.hold_body_id].copy(); hand_mat=self.data.xmat[self.hold_body_id].reshape(3,3)
        ball_pos=hand_pos+hand_mat@self.hold_relpose[:3]; ball_quat=np.empty(4,dtype=np.float64)
        mujoco.mju_mulQuat(ball_quat,self.data.xquat[self.hold_body_id],self.hold_relpose[3:7])
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr+3]=ball_pos; self.data.qpos[self.ball_qpos_adr+3:self.ball_qpos_adr+7]=ball_quat; self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6]=0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.model.nkey>0: mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        else: mujoco.mj_resetData(self.model,self.data); self.data.qpos[:]=self.nominal_qpos
        self.data.ctrl[:]=self.nominal_ctrl; self.data.qpos[self.arm_qpos_adr]+=self.np_random.uniform(-0.03,0.03,self.n_arm)
        if self.hold_eq_id>=0: self.data.eq_active[self.hold_eq_id]=1
        mujoco.mj_forward(self.model,self.data); self._place_ball_in_hand(); self.model.body_pos[self.target_body_id]=self.target_pos; mujoco.mj_forward(self.model,self.data)
        self.step_count=0; self.released=False; self.release_time=None; self.best_dist=np.inf; self.prev_action=np.zeros(self.n_arm+1)
        return self._get_obs(), {}
    def step(self, action):
        action=np.clip(np.asarray(action,dtype=np.float64),-1,1); self.data.ctrl[:]=self.nominal_ctrl
        self.data.ctrl[self.arm_actuator_ids]=self.nominal_ctrl[self.arm_actuator_ids]+self.action_scale*action[:self.n_arm]
        t=self.step_count*self.control_dt
        if not self.released:
            do_release=action[-1]>0.5 if self.learned_release else t>=self.scripted_release_time
            if do_release:
                if self.hold_eq_id>=0: self.data.eq_active[self.hold_eq_id]=0
                self.released=True; self.release_time=t
        for _ in range(self.frame_skip): mujoco.mj_step(self.model,self.data)
        self.step_count+=1; obs=self._get_obs(); reward=self._compute_reward(action); dist=np.linalg.norm(self._ball_pos()-self.target_pos); self.best_dist=min(self.best_dist,dist)
        terminated=bool(self._ball_pos()[2]<0.05 or self._ball_pos()[0]>4.0); truncated=bool(self.step_count*self.control_dt>=self.episode_time)
        info={'dist_to_target':float(dist),'best_dist':float(self.best_dist),'released':self.released,'release_time':self.release_time,'arm_joint_names':self.arm_joint_names}
        self.prev_action=action.copy(); return obs,float(reward),terminated,truncated,info
    def _compute_reward(self, action):
        ball_pos=self._ball_pos(); ball_vel=self._ball_vel(); to_target=self.target_pos-ball_pos; dist=np.linalg.norm(to_target)+1e-6; direction=to_target/dist
        vtt=np.dot(ball_vel,direction); acc=np.exp(-2.5*dist); r=0.03*max(vtt,0.0) if not self.released else 1.5*acc+0.02*max(vtt,0.0)
        if dist<0.20: r+=10
        elif dist<0.40: r+=3
        r-=0.005*np.linalg.norm(action-self.prev_action)+0.0005*np.linalg.norm(self.data.qvel[self.arm_qvel_adr])+0.0005*np.linalg.norm(action[:self.n_arm])
        return r
    def _get_obs(self):
        return np.concatenate([self.data.qpos[self.arm_qpos_adr], self.data.qvel[self.arm_qvel_adr], self._ball_pos(), self._ball_vel(), self.target_pos, self.prev_action, [1.0 if self.released else 0.0], [max(0.0,self.episode_time-self.step_count*self.control_dt)]]).astype(np.float32)
    def _ball_pos(self): return self.data.xpos[self.ball_body_id].copy()
    def _ball_vel(self):
        vel=np.zeros(6); mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.ball_body_id,vel,0); return vel[:3].copy()
