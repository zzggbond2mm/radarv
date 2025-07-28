#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于C++ SMZ系统的CV模型航迹起批实现
CV (Constant Velocity) - 恒速模型
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class TrackState(Enum):
    """航迹状态"""
    TENTATIVE = 0      # 待定（尝试）
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
    vr: float  # 径向速度
    time: float
    snr: float = 10.0
    # 原始数据
    raw_data: dict = None
    
    
class CVTrackInitiation:
    """基于CV模型的航迹起批算法"""
    
    # 常量定义（参考C++）
    MAX_FREE_TIME_INTERVAL = 20.0  # 最大时间间隔
    RAD2DEG = 180.0 / np.pi
    DEG2RAD = np.pi / 180.0
    EPS = 1e-6
    
    # 起批参数（参考C++）
    PREPARE_START_TRACK_TIMES = 3   # 预备起批所需次数
    CONFIRM_START_TRACK_TIMES = 5   # 确认起批所需次数
    MAX_CONTINUOUS_LOST_TIMES = 3   # 最大连续丢失次数
    
    def __init__(self, 
                 prepare_times=3,
                 confirm_times=5, 
                 max_lost_times=3,
                 correlate_range_threshold=200.0,  # 距离关联门限（米）
                 correlate_azim_threshold=5.0,     # 方位关联门限（度）
                 correlate_velocity_threshold=10.0):  # 速度关联门限（米/秒）
        
        self.prepare_times = prepare_times
        self.confirm_times = confirm_times
        self.max_lost_times = max_lost_times
        
        # 关联门限
        self.correlate_range_threshold = correlate_range_threshold
        self.correlate_azim_threshold = correlate_azim_threshold
        self.correlate_velocity_threshold = correlate_velocity_threshold
        
        # 航迹存储
        self.tracks = {}
        self.next_track_id = 1
        
        # 统计信息
        self.stats = {
            'total_created': 0,
            'tentative': 0,
            'prepare': 0,
            'confirmed': 0,
            'terminated': 0
        }
        
    def process_frame(self, detections: pd.DataFrame, current_time: float) -> Dict:
        """处理一帧数据"""
        if detections is None or detections.empty:
            # 无检测，所有航迹外推
            self._update_all_tracks_no_detection(current_time)
            return self.tracks
            
        # 1. 生成所有航迹的预测
        self._generate_predictions(current_time)
        
        # 2. 点航迹关联
        associations = self._correlate_points_to_tracks(detections)
        
        # 3. 更新关联上的航迹
        self._update_associated_tracks(associations, detections, current_time)
        
        # 4. 更新未关联的航迹（外推）
        self._update_unassociated_tracks(current_time)
        
        # 5. 起始新航迹
        self._start_new_tracks(detections, associations, current_time)
        
        # 6. 更新航迹状态
        self._update_track_states()
        
        # 7. 删除终止的航迹
        self._remove_terminated_tracks()
        
        # 8. 更新统计
        self._update_statistics()
        
        return self.tracks
        
    def _generate_predictions(self, current_time: float):
        """生成CV模型预测（参考C++的generateNextPrediction）"""
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if not track['points']:
                continue
                
            # 获取最后一个点
            last_point = track['points'][-1]
            delta_time = current_time - last_point.time
            
            # 时间间隔检查
            if delta_time < 0 or delta_time > self.MAX_FREE_TIME_INTERVAL:
                track['prediction'] = None
                continue
                
            # CV模型预测：假设速度恒定
            pred = TrackPoint(
                x = last_point.x + last_point.vx * delta_time,
                y = last_point.y + last_point.vy * delta_time,
                z = max(0, last_point.z + last_point.vz * delta_time),  # 高度不能为负
                vx = last_point.vx,
                vy = last_point.vy,
                vz = last_point.vz,
                range = 0,  # 后续计算
                azim_deg = 0,
                elev_deg = 0,
                vr = 0,
                time = current_time
            )
            
            # 计算极坐标参数
            pred.range = np.sqrt(pred.x**2 + pred.y**2 + pred.z**2)
            pred.azim_deg = np.arctan2(pred.x, -pred.y) * self.RAD2DEG
            if pred.azim_deg < 0:
                pred.azim_deg += 360
                
            if pred.range > self.EPS:
                pred.elev_deg = np.arcsin(pred.z / pred.range) * self.RAD2DEG
                # 径向速度 = 速度向量在径向上的投影（取反）
                pred.vr = -(pred.vx * pred.x + pred.vy * pred.y + pred.vz * pred.z) / pred.range
                
            track['prediction'] = pred
            
    def _correlate_points_to_tracks(self, detections: pd.DataFrame) -> Dict[int, int]:
        """点航迹关联（参考C++的correlatePointsToTracks）"""
        associations = {}
        used_detections = set()
        
        # 按航迹质量排序（确认的优先）
        sorted_tracks = sorted(self.tracks.items(), 
                             key=lambda x: (x[1]['state'].value, -len(x[1]['points'])),
                             reverse=True)
        
        for track_id, track in sorted_tracks:
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if track['prediction'] is None:
                continue
                
            pred = track['prediction']
            best_score = float('inf')
            best_idx = -1
            
            # 遍历所有检测点
            for idx, det in detections.iterrows():
                if idx in used_detections:
                    continue
                    
                # 转换检测点到笛卡尔坐标
                r = det['range_out']
                azim_rad = det['azim_out'] * self.DEG2RAD
                x = r * np.sin(azim_rad)
                y = -r * np.cos(azim_rad)
                z = det.get('high', 0)
                
                # 计算关联分数
                # 1. 距离差
                dr = np.sqrt((x - pred.x)**2 + (y - pred.y)**2 + (z - pred.z)**2)
                
                # 2. 方位差
                det_azim = det['azim_out']
                da = abs(det_azim - pred.azim_deg)
                if da > 180:
                    da = 360 - da
                    
                # 3. 速度差
                dv = abs(det.get('v_out', 0) - pred.vr)
                
                # 检查门限
                if (dr <= self.correlate_range_threshold and
                    da <= self.correlate_azim_threshold and
                    dv <= self.correlate_velocity_threshold):
                    
                    # 计算综合分数（归一化）
                    score = (dr / self.correlate_range_threshold + 
                            da / self.correlate_azim_threshold + 
                            dv / self.correlate_velocity_threshold)
                    
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        
            # 关联最佳匹配
            if best_idx >= 0:
                associations[track_id] = best_idx
                used_detections.add(best_idx)
                
        return associations
        
    def _update_associated_tracks(self, associations: Dict[int, int], 
                                 detections: pd.DataFrame, current_time: float):
        """更新关联上的航迹"""
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            det = detections.iloc[det_idx]
            
            # 转换到笛卡尔坐标
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 计算速度（如果有多个历史点）
            if len(track['points']) >= 1:
                last_point = track['points'][-1]
                dt = current_time - last_point.time
                
                if dt > 0 and dt < 5.0:  # 合理的时间间隔
                    vx = (x - last_point.x) / dt
                    vy = (y - last_point.y) / dt
                    vz = (z - last_point.z) / dt
                else:
                    # 使用径向速度估算
                    vr = det.get('v_out', 0)
                    if r > 0:
                        vx = vr * x / r
                        vy = vr * y / r
                        vz = 0
                    else:
                        vx = vy = vz = 0
            else:
                # 第一个点，用径向速度估算
                vr = det.get('v_out', 0)
                if r > 0:
                    vx = vr * x / r
                    vy = vr * y / r
                    vz = 0
                else:
                    vx = vy = vz = 0
                    
            # 创建新的航迹点
            new_point = TrackPoint(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
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
                
    def _update_unassociated_tracks(self, current_time: float):
        """更新未关联的航迹（外推）"""
        for track_id, track in self.tracks.items():
            if track.get('is_associated', False):
                track['is_associated'] = False  # 重置标志
                continue
                
            if track['state'] == TrackState.TERMINATED:
                continue
                
            # 增加丢失次数
            track['consecutive_lost'] += 1
            
            # 如果有预测，添加外推点
            if track['prediction'] is not None:
                track['points'].append(track['prediction'])
                track['last_update_time'] = current_time
                
    def _start_new_tracks(self, detections: pd.DataFrame, 
                         associations: Dict[int, int], current_time: float):
        """起始新航迹"""
        used_indices = set(associations.values())
        
        for idx, det in detections.iterrows():
            if idx in used_indices:
                continue
                
            # 过滤条件（可选）
            if det.get('SNR/10', 0) < 10:  # SNR太低
                continue
                
            # 创建新航迹
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # 转换到笛卡尔坐标
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 初始速度估算
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
                snr=det.get('SNR/10', 10),
                raw_data=det.to_dict()
            )
            
            # 创建新航迹
            self.tracks[track_id] = {
                'track_id': track_id,
                'points': [first_point],
                'state': TrackState.TENTATIVE,
                'track_times': 1,
                'consecutive_lost': 0,
                'created_time': current_time,
                'last_update_time': current_time,
                'prediction': None,
                'is_associated': False
            }
            
            self.stats['total_created'] += 1
            
    def _update_track_states(self):
        """更新航迹状态（参考C++逻辑）"""
        for track in self.tracks.values():
            # 检查是否应该终止
            if track['consecutive_lost'] > self.max_lost_times:
                track['state'] = TrackState.TERMINATED
                continue
                
            # 根据跟踪次数更新状态
            track_times = track['track_times']
            
            if track_times >= self.confirm_times:
                if track['state'] != TrackState.CONFIRMED:
                    print(f"航迹 T{track['track_id']} 确认起批")
                track['state'] = TrackState.CONFIRMED
            elif track_times >= self.prepare_times:
                if track['state'] == TrackState.TENTATIVE:
                    print(f"航迹 T{track['track_id']} 预备起批")
                track['state'] = TrackState.PREPARE
                
    def _remove_terminated_tracks(self):
        """删除终止的航迹"""
        terminated_ids = [tid for tid, track in self.tracks.items() 
                         if track['state'] == TrackState.TERMINATED]
        
        for tid in terminated_ids:
            print(f"删除终止航迹 T{tid}")
            del self.tracks[tid]
            
    def _update_all_tracks_no_detection(self, current_time: float):
        """无检测时更新所有航迹"""
        for track in self.tracks.values():
            track['consecutive_lost'] += 1
            track['is_associated'] = False
            
    def _update_statistics(self):
        """更新统计信息"""
        self.stats['tentative'] = sum(1 for t in self.tracks.values() 
                                     if t['state'] == TrackState.TENTATIVE)
        self.stats['prepare'] = sum(1 for t in self.tracks.values() 
                                   if t['state'] == TrackState.PREPARE)
        self.stats['confirmed'] = sum(1 for t in self.tracks.values() 
                                     if t['state'] == TrackState.CONFIRMED)
                                     
    def get_display_tracks(self):
        """获取用于显示的航迹"""
        display_tracks = {}
        
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            # 至少需要2个点画线
            if len(track['points']) < 2:
                continue
                
            # 转换点格式
            points = []
            for pt in track['points']:
                points.append({
                    'x': pt.x,
                    'y': pt.y,
                    'range_out': pt.range,
                    'azim_out': pt.azim_deg
                })
                
            # 状态映射
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