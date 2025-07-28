#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV模型 + 卡尔曼滤波的航迹起批
结合恒速运动模型和卡尔曼滤波器进行状态估计和预测
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class TrackState(Enum):
    """航迹状态"""
    TENTATIVE = 0      # 待定
    PREPARE = 1        # 预备起批
    CONFIRMED = 2      # 确认起批  
    TERMINATED = -1    # 终止


@dataclass
class TrackPoint:
    """航迹点数据结构"""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    range: float
    azim_deg: float
    elev_deg: float
    vr: float
    time: float
    snr: float = 10.0
    raw_data: dict = None


class KalmanFilter:
    """卡尔曼滤波器 - CV模型（恒速模型）"""
    
    def __init__(self, dt=1.0, process_noise_std=5.0, measurement_noise_std=10.0):
        """
        初始化卡尔曼滤波器
        dt: 时间步长
        process_noise_std: 过程噪声标准差（米/秒²）
        measurement_noise_std: 测量噪声标准差（米）
        """
        self.dt = dt
        self.initialized = False
        
        # 状态向量: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # 状态转移矩阵 F (CV模型)
        self.F = np.eye(6)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt
        
        # 观测矩阵 H (只观测位置)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1  # 观测x
        self.H[1, 1] = 1  # 观测y
        self.H[2, 2] = 1  # 观测z
        
        # 过程噪声协方差矩阵 Q
        q = process_noise_std ** 2
        self.Q = np.zeros((6, 6))
        # 位置的过程噪声
        self.Q[0, 0] = (dt**4)/4 * q
        self.Q[1, 1] = (dt**4)/4 * q
        self.Q[2, 2] = (dt**4)/4 * q
        # 速度的过程噪声
        self.Q[3, 3] = (dt**2) * q
        self.Q[4, 4] = (dt**2) * q
        self.Q[5, 5] = (dt**2) * q
        # 位置-速度协方差
        self.Q[0, 3] = (dt**3)/2 * q
        self.Q[3, 0] = (dt**3)/2 * q
        self.Q[1, 4] = (dt**3)/2 * q
        self.Q[4, 1] = (dt**3)/2 * q
        self.Q[2, 5] = (dt**3)/2 * q
        self.Q[5, 2] = (dt**3)/2 * q
        
        # 测量噪声协方差矩阵 R
        r = measurement_noise_std ** 2
        self.R = np.eye(3) * r
        
        # 误差协方差矩阵 P
        self.P = np.eye(6) * 100.0
        
    def initialize(self, x, y, z, vx=0, vy=0, vz=0):
        """初始化滤波器状态"""
        self.state = np.array([x, y, z, vx, vy, vz])
        self.P = np.eye(6) * 100.0
        # 速度不确定性更大
        self.P[3, 3] = 1000.0
        self.P[4, 4] = 1000.0
        self.P[5, 5] = 1000.0
        self.initialized = True
        
    def predict(self, dt):
        """预测步骤"""
        if not self.initialized:
            return
            
        # 更新状态转移矩阵的时间步长
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # 更新过程噪声（根据时间步长）
        q = 5.0 ** 2  # 过程噪声强度
        self.Q[0, 0] = (dt**4)/4 * q
        self.Q[1, 1] = (dt**4)/4 * q
        self.Q[2, 2] = (dt**4)/4 * q
        self.Q[3, 3] = (dt**2) * q
        self.Q[4, 4] = (dt**2) * q
        self.Q[5, 5] = (dt**2) * q
        self.Q[0, 3] = self.Q[3, 0] = (dt**3)/2 * q
        self.Q[1, 4] = self.Q[4, 1] = (dt**3)/2 * q
        self.Q[2, 5] = self.Q[5, 2] = (dt**3)/2 * q
        
        # 状态预测
        self.state = self.F @ self.state
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z_measured):
        """更新步骤"""
        if not self.initialized:
            return
            
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = z_measured - self.H @ self.state  # 测量残差
        self.state = self.state + K @ y
        
        # 协方差更新
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
    def get_state(self):
        """获取当前状态"""
        return self.state.copy()
        
    def get_position(self):
        """获取位置"""
        return self.state[:3]
        
    def get_velocity(self):
        """获取速度"""
        return self.state[3:6]


class CVKalmanTrackInitiation:
    """CV模型 + 卡尔曼滤波的航迹起批"""
    
    # 常量定义
    MAX_FREE_TIME_INTERVAL = 20.0
    RAD2DEG = 180.0 / np.pi
    DEG2RAD = np.pi / 180.0
    EPS = 1e-6
    
    def __init__(self):
        # 起批参数
        self.prepare_times = 2      # 2次即可预备
        self.confirm_times = 3      # 3次即可确认
        self.max_lost_times = 5     # 5次才删除
        
        # 关联门限
        self.base_range_threshold = 300.0    # 基础距离门限（米）
        self.base_azim_threshold = 8.0       # 基础方位门限（度）
        self.base_velocity_threshold = 20.0  # 基础速度门限（米/秒）
        
        # 航迹合并门限
        self.merge_range_threshold = 150.0   
        self.merge_velocity_threshold = 15.0 
        
        # 航迹存储
        self.tracks = {}
        self.next_track_id = 1
        
        # 统计信息
        self.stats = {
            'total_created': 0,
            'tentative': 0,
            'prepare': 0,
            'confirmed': 0,
            'terminated': 0,
            'merged': 0
        }
        
    def process_frame(self, detections: pd.DataFrame, current_time: float) -> Dict:
        """处理一帧数据"""
        if detections is None or detections.empty:
            self._update_all_tracks_no_detection(current_time)
            return self.tracks
            
        # 1. 使用卡尔曼滤波器预测所有航迹
        self._kalman_predict_all_tracks(current_time)
        
        # 2. 点航迹关联
        associations = self._associate_detections_to_tracks(detections, current_time)
        
        # 3. 使用卡尔曼滤波器更新关联的航迹
        self._kalman_update_associated_tracks(associations, detections, current_time)
        
        # 4. 更新未关联的航迹
        self._update_unassociated_tracks(current_time)
        
        # 5. 起始新航迹
        self._start_new_tracks(detections, associations, current_time)
        
        # 6. 合并相似航迹
        self._merge_similar_tracks()
        
        # 7. 更新航迹状态
        self._update_track_states()
        
        # 8. 删除终止的航迹
        self._remove_terminated_tracks()
        
        # 9. 更新统计
        self._update_statistics()
        
        return self.tracks
        
    def _kalman_predict_all_tracks(self, current_time: float):
        """使用卡尔曼滤波器预测所有航迹"""
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            kalman = track.get('kalman_filter')
            if kalman is None or not kalman.initialized:
                continue
                
            # 计算时间间隔
            last_time = track.get('last_update_time', current_time)
            dt = current_time - last_time
            
            if dt <= 0 or dt > self.MAX_FREE_TIME_INTERVAL:
                continue
                
            # 卡尔曼预测
            kalman.predict(dt)
            
            # 获取预测状态
            state = kalman.get_state()
            
            # 创建预测点
            pred = TrackPoint(
                x=state[0],
                y=state[1], 
                z=max(0, state[2]),  # 高度不能为负
                vx=state[3],
                vy=state[4],
                vz=state[5],
                range=0,
                azim_deg=0,
                elev_deg=0,
                vr=0,
                time=current_time
            )
            
            # 计算极坐标
            pred.range = np.sqrt(pred.x**2 + pred.y**2 + pred.z**2)
            pred.azim_deg = np.arctan2(pred.x, -pred.y) * self.RAD2DEG
            if pred.azim_deg < 0:
                pred.azim_deg += 360
                
            if pred.range > self.EPS:
                pred.elev_deg = np.arcsin(pred.z / pred.range) * self.RAD2DEG
                pred.vr = -(pred.vx * pred.x + pred.vy * pred.y + pred.vz * pred.z) / pred.range
                
            track['prediction'] = pred
            
    def _associate_detections_to_tracks(self, detections: pd.DataFrame, 
                                      current_time: float) -> Dict[int, int]:
        """关联检测点到航迹"""
        associations = {}
        used_detections = set()
        
        # 重置索引，确保索引从0开始连续
        detections = detections.reset_index(drop=True)
        
        # 计算每个检测点的笛卡尔坐标
        det_coords = []
        for idx, det in detections.iterrows():
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            vr = det.get('v_out', 0)
            det_coords.append((idx, x, y, z, vr, det))
            
        # 按航迹质量排序
        sorted_tracks = sorted(self.tracks.items(), 
                             key=lambda x: (x[1]['state'].value, 
                                          len(x[1]['points']), 
                                          -x[1]['consecutive_lost']),
                             reverse=True)
        
        for track_id, track in sorted_tracks:
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if track.get('prediction') is None:
                continue
                
            pred = track['prediction']
            best_score = float('inf')
            best_idx = -1
            
            # 动态调整门限
            lost_factor = 1.0 + track['consecutive_lost'] * 0.5
            
            range_threshold = self.base_range_threshold * lost_factor
            azim_threshold = self.base_azim_threshold * lost_factor
            velocity_threshold = self.base_velocity_threshold * lost_factor
            
            # 使用马氏距离（考虑不确定性）
            kalman = track.get('kalman_filter')
            if kalman and kalman.initialized:
                # 获取位置协方差
                P_pos = kalman.P[:3, :3]
                
            # 遍历所有未关联的检测点
            for idx, x, y, z, vr, det in det_coords:
                if idx in used_detections:
                    continue
                    
                # 计算关联分数
                dr = np.sqrt((x - pred.x)**2 + (y - pred.y)**2 + (z - pred.z)**2)
                
                # 方位差
                det_azim = det['azim_out']
                da = abs(det_azim - pred.azim_deg)
                if da > 180:
                    da = 360 - da
                    
                # 速度差
                dv = abs(vr - pred.vr)
                
                # 检查门限
                if (dr <= range_threshold and
                    da <= azim_threshold and
                    dv <= velocity_threshold):
                    
                    # 计算加权分数
                    range_weight = 1.0
                    azim_weight = 0.5
                    velocity_weight = 0.3
                    
                    score = (range_weight * dr / range_threshold + 
                            azim_weight * da / azim_threshold + 
                            velocity_weight * dv / velocity_threshold)
                    
                    # 考虑SNR
                    snr = det.get('SNR/10', 10)
                    snr_factor = min(1.0, snr / 20.0)
                    score *= (2.0 - snr_factor)
                    
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        
            # 关联最佳匹配
            if best_idx >= 0 and best_score < 2.0:
                associations[track_id] = best_idx
                used_detections.add(best_idx)
                
        return associations
        
    def _kalman_update_associated_tracks(self, associations: Dict[int, int], 
                                        detections: pd.DataFrame, current_time: float):
        """使用卡尔曼滤波器更新关联的航迹"""
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            det = detections.iloc[det_idx]
            
            # 转换坐标
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 获取或创建卡尔曼滤波器
            kalman = track.get('kalman_filter')
            if kalman is None:
                kalman = KalmanFilter()
                track['kalman_filter'] = kalman
                
            # 如果是第一次，初始化卡尔曼滤波器
            if not kalman.initialized:
                # 估计初始速度
                vr = det.get('v_out', 0)
                if r > 0:
                    vx = vr * x / r
                    vy = vr * y / r
                    vz = 0
                else:
                    vx = vy = vz = 0
                    
                kalman.initialize(x, y, z, vx, vy, vz)
            else:
                # 卡尔曼更新
                z_measured = np.array([x, y, z])
                kalman.update(z_measured)
                
            # 获取滤波后的状态
            state = kalman.get_state()
            
            # 创建新点（使用滤波后的值）
            new_point = TrackPoint(
                x=state[0], 
                y=state[1], 
                z=state[2],
                vx=state[3], 
                vy=state[4], 
                vz=state[5],
                range=r,
                azim_deg=det['azim_out'],
                elev_deg=det.get('elev1', 0),
                vr=det.get('v_out', 0),
                time=current_time,
                snr=det.get('SNR/10', 10),
                raw_data=det.to_dict()
            )
            
            # 更新航迹
            track['points'].append(new_point)
            track['consecutive_lost'] = 0
            track['track_times'] += 1
            track['last_update_time'] = current_time
            track['is_associated'] = True
            
            # 限制历史长度
            if len(track['points']) > 50:
                track['points'] = track['points'][-50:]
                
    def _merge_similar_tracks(self):
        """合并相似的航迹"""
        if len(self.tracks) < 2:
            return
            
        merge_candidates = []
        track_ids = list(self.tracks.keys())
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = self.tracks[track_ids[i]]
                track2 = self.tracks[track_ids[j]]
                
                if (track1['state'] == TrackState.TERMINATED or 
                    track2['state'] == TrackState.TERMINATED):
                    continue
                    
                # 使用卡尔曼滤波器的状态进行比较
                kalman1 = track1.get('kalman_filter')
                kalman2 = track2.get('kalman_filter')
                
                if kalman1 and kalman2 and kalman1.initialized and kalman2.initialized:
                    state1 = kalman1.get_state()
                    state2 = kalman2.get_state()
                    
                    # 位置差
                    dr = np.sqrt((state1[0] - state2[0])**2 + 
                               (state1[1] - state2[1])**2 + 
                               (state1[2] - state2[2])**2)
                    
                    # 速度差
                    dv = np.sqrt((state1[3] - state2[3])**2 + 
                               (state1[4] - state2[4])**2 + 
                               (state1[5] - state2[5])**2)
                    
                    if (dr < self.merge_range_threshold and 
                        dv < self.merge_velocity_threshold):
                        
                        # 优先保留质量更好的航迹
                        if (track1['state'].value > track2['state'].value or
                            (track1['state'] == track2['state'] and 
                             len(track1['points']) > len(track2['points']))):
                            merge_candidates.append((track_ids[i], track_ids[j]))
                        else:
                            merge_candidates.append((track_ids[j], track_ids[i]))
                            
        # 执行合并
        for keep_id, merge_id in merge_candidates:
            if keep_id in self.tracks and merge_id in self.tracks:
                self._merge_tracks(keep_id, merge_id)
                
    def _merge_tracks(self, keep_id: int, merge_id: int):
        """合并两条航迹"""
        keep_track = self.tracks[keep_id]
        merge_track = self.tracks[merge_id]
        
        # 合并点
        all_points = keep_track['points'] + merge_track['points']
        all_points.sort(key=lambda p: p.time)
        
        # 去重
        filtered_points = []
        for point in all_points:
            if not filtered_points or point.time - filtered_points[-1].time > 0.1:
                filtered_points.append(point)
                
        keep_track['points'] = filtered_points[-50:]
        
        # 保留更好的卡尔曼滤波器状态
        kalman1 = keep_track.get('kalman_filter')
        kalman2 = merge_track.get('kalman_filter')
        
        if kalman2 and (not kalman1 or 
                       (kalman2.initialized and np.trace(kalman2.P) < np.trace(kalman1.P))):
            keep_track['kalman_filter'] = kalman2
            
        # 更新跟踪次数
        keep_track['track_times'] = max(keep_track['track_times'], 
                                       merge_track['track_times'])
        
        # 删除被合并的航迹
        del self.tracks[merge_id]
        self.stats['merged'] += 1
        print(f"合并航迹: T{merge_id} -> T{keep_id}")
        
    def _update_track_states(self):
        """更新航迹状态"""
        for track in self.tracks.values():
            if track['consecutive_lost'] > self.max_lost_times:
                track['state'] = TrackState.TERMINATED
                continue
                
            track_times = track['track_times']
            
            if track_times >= self.confirm_times:
                if track['state'] != TrackState.CONFIRMED:
                    print(f"航迹 T{track['track_id']} 确认起批 (点数: {len(track['points'])})")
                track['state'] = TrackState.CONFIRMED
            elif track_times >= self.prepare_times:
                if track['state'] == TrackState.TENTATIVE:
                    print(f"航迹 T{track['track_id']} 预备起批 (点数: {len(track['points'])})")
                track['state'] = TrackState.PREPARE
                
    def _start_new_tracks(self, detections: pd.DataFrame, 
                         associations: Dict[int, int], current_time: float):
        """起始新航迹"""
        used_indices = set(associations.values())
        
        for idx, det in detections.iterrows():
            if idx in used_indices:
                continue
                
            # 质量过滤
            snr = det.get('SNR/10', 0)
            if snr < 8:
                continue
                
            # 坐标转换
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 检查是否离现有航迹太近
            too_close = False
            for track in self.tracks.values():
                if track['points'] and track['state'] != TrackState.TERMINATED:
                    kalman = track.get('kalman_filter')
                    if kalman and kalman.initialized:
                        state = kalman.get_state()
                        dist = np.sqrt((x - state[0])**2 + (y - state[1])**2)
                        if dist < 100:
                            too_close = True
                            break
                            
            if too_close:
                continue
                
            # 创建新航迹
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # 估计初始速度
            vr = det.get('v_out', 0)
            if r > 0:
                vx = vr * x / r
                vy = vr * y / r
                vz = 0
            else:
                vx = vy = vz = 0
                
            # 创建第一个点
            first_point = TrackPoint(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                range=r,
                azim_deg=det['azim_out'],
                elev_deg=det.get('elev1', 0),
                vr=vr,
                time=current_time,
                snr=snr,
                raw_data=det.to_dict()
            )
            
            # 创建卡尔曼滤波器
            kalman = KalmanFilter()
            kalman.initialize(x, y, z, vx, vy, vz)
            
            # 创建航迹
            self.tracks[track_id] = {
                'track_id': track_id,
                'points': [first_point],
                'state': TrackState.TENTATIVE,
                'track_times': 1,
                'consecutive_lost': 0,
                'created_time': current_time,
                'last_update_time': current_time,
                'prediction': None,
                'is_associated': False,
                'kalman_filter': kalman
            }
            
            self.stats['total_created'] += 1
            
    def _update_unassociated_tracks(self, current_time: float):
        """更新未关联的航迹"""
        for track_id, track in self.tracks.items():
            if track.get('is_associated', False):
                track['is_associated'] = False
                continue
                
            if track['state'] == TrackState.TERMINATED:
                continue
                
            track['consecutive_lost'] += 1
            
            # 对于确认的航迹，使用卡尔曼预测进行外推
            if (track['prediction'] is not None and 
                track['state'] == TrackState.CONFIRMED and
                track.get('kalman_filter')):
                
                track['points'].append(track['prediction'])
                track['last_update_time'] = current_time
                
    def _update_all_tracks_no_detection(self, current_time: float):
        """无检测时更新"""
        for track in self.tracks.values():
            track['consecutive_lost'] += 1
            track['is_associated'] = False
            
    def _remove_terminated_tracks(self):
        """删除终止的航迹"""
        terminated_ids = [tid for tid, track in self.tracks.items() 
                         if track['state'] == TrackState.TERMINATED]
        
        for tid in terminated_ids:
            print(f"删除终止航迹 T{tid}")
            del self.tracks[tid]
            
    def _update_statistics(self):
        """更新统计"""
        self.stats['tentative'] = sum(1 for t in self.tracks.values() 
                                     if t['state'] == TrackState.TENTATIVE)
        self.stats['prepare'] = sum(1 for t in self.tracks.values() 
                                   if t['state'] == TrackState.PREPARE)
        self.stats['confirmed'] = sum(1 for t in self.tracks.values() 
                                     if t['state'] == TrackState.CONFIRMED)
                                     
    def get_display_tracks(self):
        """获取显示用的航迹"""
        display_tracks = {}
        
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            # 至少2个点
            if len(track['points']) < 2:
                continue
                
            points = []
            for pt in track['points']:
                points.append({
                    'x': pt.x,
                    'y': pt.y,
                    'range_out': pt.range,
                    'azim_out': pt.azim_deg
                })
                
            mode_map = {
                TrackState.TENTATIVE: 0,
                TrackState.PREPARE: 1,
                TrackState.CONFIRMED: 2
            }
            
            display_tracks[track_id] = {
                'points': points,
                'established_mode': mode_map.get(track['state'], 0),
                'track_times': track['track_times'],
                'consecutive_lost': track['consecutive_lost']
            }
            
        return display_tracks