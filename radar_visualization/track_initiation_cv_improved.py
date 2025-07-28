#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的CV模型航迹起批 - 增强关联能力
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
    

class ImprovedCVTrackInitiation:
    """改进的CV模型航迹起批 - 更好的关联和融合"""
    
    # 常量定义
    MAX_FREE_TIME_INTERVAL = 20.0
    RAD2DEG = 180.0 / np.pi
    DEG2RAD = np.pi / 180.0
    EPS = 1e-6
    
    def __init__(self):
        # 起批参数
        self.prepare_times = 2      # 降低到2次即可预备
        self.confirm_times = 3      # 降低到3次即可确认
        self.max_lost_times = 5     # 增加到5次才删除
        
        # 关联门限 - 动态调整
        self.base_range_threshold = 300.0    # 基础距离门限（米）
        self.base_azim_threshold = 8.0       # 基础方位门限（度）
        self.base_velocity_threshold = 20.0  # 基础速度门限（米/秒）
        
        # 融合门限 - 用于判断是否应该合并航迹
        self.merge_range_threshold = 150.0   # 航迹合并距离门限
        self.merge_velocity_threshold = 15.0 # 航迹合并速度门限
        
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
            
        # 1. 生成预测
        self._generate_predictions(current_time)
        
        # 2. 点航迹关联 - 改进的关联算法
        associations = self._improved_correlate_points_to_tracks(detections, current_time)
        
        # 3. 更新关联的航迹
        self._update_associated_tracks(associations, detections, current_time)
        
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
        
    def _generate_predictions(self, current_time: float):
        """生成CV模型预测 - 增加预测可信度"""
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if not track['points']:
                continue
                
            # 使用最近N个点计算平均速度（更稳定）
            n_points = min(5, len(track['points']))
            if n_points >= 2:
                # 计算平均速度
                total_vx = total_vy = total_vz = 0
                valid_count = 0
                
                for i in range(1, n_points):
                    p1 = track['points'][-i-1]
                    p2 = track['points'][-i]
                    dt = p2.time - p1.time
                    
                    if 0 < dt < 2.0:  # 合理的时间间隔
                        vx = (p2.x - p1.x) / dt
                        vy = (p2.y - p1.y) / dt
                        vz = (p2.z - p1.z) / dt
                        
                        total_vx += vx
                        total_vy += vy
                        total_vz += vz
                        valid_count += 1
                        
                if valid_count > 0:
                    avg_vx = total_vx / valid_count
                    avg_vy = total_vy / valid_count
                    avg_vz = total_vz / valid_count
                else:
                    # 使用最后一个点的速度
                    last_point = track['points'][-1]
                    avg_vx = last_point.vx
                    avg_vy = last_point.vy
                    avg_vz = last_point.vz
            else:
                # 只有一个点，使用其速度
                last_point = track['points'][-1]
                avg_vx = last_point.vx
                avg_vy = last_point.vy
                avg_vz = last_point.vz
                
            # 预测
            last_point = track['points'][-1]
            delta_time = current_time - last_point.time
            
            if delta_time < 0 or delta_time > self.MAX_FREE_TIME_INTERVAL:
                track['prediction'] = None
                continue
                
            pred = TrackPoint(
                x = last_point.x + avg_vx * delta_time,
                y = last_point.y + avg_vy * delta_time,
                z = max(0, last_point.z + avg_vz * delta_time),
                vx = avg_vx,
                vy = avg_vy,
                vz = avg_vz,
                range = 0,
                azim_deg = 0,
                elev_deg = 0,
                vr = 0,
                time = current_time
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
            
    def _improved_correlate_points_to_tracks(self, detections: pd.DataFrame, 
                                           current_time: float) -> Dict[int, int]:
        """改进的点航迹关联 - 考虑航迹质量和历史"""
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
            
        # 按航迹质量排序（确认>预备>待定，点数多的优先）
        sorted_tracks = sorted(self.tracks.items(), 
                             key=lambda x: (x[1]['state'].value, 
                                          len(x[1]['points']), 
                                          -x[1]['consecutive_lost']),
                             reverse=True)
        
        for track_id, track in sorted_tracks:
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if track['prediction'] is None:
                continue
                
            pred = track['prediction']
            best_score = float('inf')
            best_idx = -1
            
            # 动态调整门限
            track_quality = min(1.0, len(track['points']) / 10.0)
            lost_factor = 1.0 + track['consecutive_lost'] * 0.5
            
            range_threshold = self.base_range_threshold * lost_factor
            azim_threshold = self.base_azim_threshold * lost_factor
            velocity_threshold = self.base_velocity_threshold * lost_factor
            
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
                    
                    # 考虑SNR作为额外权重
                    snr = det.get('SNR/10', 10)
                    snr_factor = min(1.0, snr / 20.0)
                    score *= (2.0 - snr_factor)  # SNR越高，分数越低（越好）
                    
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        
            # 关联最佳匹配
            if best_idx >= 0 and best_score < 2.0:  # 添加最大分数限制
                associations[track_id] = best_idx
                used_detections.add(best_idx)
                
        return associations
        
    def _merge_similar_tracks(self):
        """合并相似的航迹"""
        if len(self.tracks) < 2:
            return
            
        # 找出所有可能需要合并的航迹对
        merge_candidates = []
        track_ids = list(self.tracks.keys())
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = self.tracks[track_ids[i]]
                track2 = self.tracks[track_ids[j]]
                
                # 跳过终止的航迹
                if (track1['state'] == TrackState.TERMINATED or 
                    track2['state'] == TrackState.TERMINATED):
                    continue
                    
                # 检查是否应该合并
                if self._should_merge_tracks(track1, track2):
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
                
    def _should_merge_tracks(self, track1, track2) -> bool:
        """判断两条航迹是否应该合并"""
        if not track1['points'] or not track2['points']:
            return False
            
        # 比较最新点的位置和速度
        p1 = track1['points'][-1]
        p2 = track2['points'][-1]
        
        # 位置差
        dr = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        
        # 速度差
        dv = np.sqrt((p1.vx - p2.vx)**2 + (p1.vy - p2.vy)**2 + (p1.vz - p2.vz)**2)
        
        # 时间差
        dt = abs(p1.time - p2.time)
        
        # 判断条件
        if (dr < self.merge_range_threshold and 
            dv < self.merge_velocity_threshold and
            dt < 5.0):  # 时间相近
            return True
            
        return False
        
    def _merge_tracks(self, keep_id: int, merge_id: int):
        """合并两条航迹"""
        keep_track = self.tracks[keep_id]
        merge_track = self.tracks[merge_id]
        
        # 合并点（按时间排序）
        all_points = keep_track['points'] + merge_track['points']
        all_points.sort(key=lambda p: p.time)
        
        # 去重（时间相近的点只保留一个）
        filtered_points = []
        for point in all_points:
            if not filtered_points or point.time - filtered_points[-1].time > 0.1:
                filtered_points.append(point)
                
        keep_track['points'] = filtered_points[-50:]  # 保留最近50个点
        
        # 更新跟踪次数
        keep_track['track_times'] = max(keep_track['track_times'], 
                                       merge_track['track_times'])
        
        # 删除被合并的航迹
        del self.tracks[merge_id]
        self.stats['merged'] += 1
        print(f"合并航迹: T{merge_id} -> T{keep_id}")
        
    def _update_associated_tracks(self, associations: Dict[int, int], 
                                 detections: pd.DataFrame, current_time: float):
        """更新关联上的航迹 - 改进速度估算"""
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            det = detections.iloc[det_idx]
            
            # 转换坐标
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 速度估算 - 使用卡尔曼滤波思想进行平滑
            if len(track['points']) >= 1:
                last_point = track['points'][-1]
                dt = current_time - last_point.time
                
                if 0 < dt < 2.0:
                    # 计算瞬时速度
                    inst_vx = (x - last_point.x) / dt
                    inst_vy = (y - last_point.y) / dt
                    inst_vz = (z - last_point.z) / dt
                    
                    # 速度平滑（避免突变）
                    alpha = 0.7  # 平滑因子
                    vx = alpha * inst_vx + (1 - alpha) * last_point.vx
                    vy = alpha * inst_vy + (1 - alpha) * last_point.vy
                    vz = alpha * inst_vz + (1 - alpha) * last_point.vz
                else:
                    # 使用径向速度
                    vr = det.get('v_out', 0)
                    if r > 0:
                        vx = vr * x / r
                        vy = vr * y / r
                        vz = 0
                    else:
                        vx = vy = vz = 0
            else:
                # 第一个点
                vr = det.get('v_out', 0)
                if r > 0:
                    vx = vr * x / r
                    vy = vr * y / r
                    vz = 0
                else:
                    vx = vy = vz = 0
                    
            # 创建新点
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
                
    def _update_track_states(self):
        """更新航迹状态 - 更快的状态转换"""
        for track in self.tracks.values():
            if track['consecutive_lost'] > self.max_lost_times:
                track['state'] = TrackState.TERMINATED
                continue
                
            track_times = track['track_times']
            
            # 更快的状态提升
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
        """起始新航迹 - 更严格的筛选"""
        used_indices = set(associations.values())
        
        for idx, det in detections.iterrows():
            if idx in used_indices:
                continue
                
            # 质量过滤
            snr = det.get('SNR/10', 0)
            if snr < 8:  # 提高SNR门限
                continue
                
            # 检查是否离现有航迹太近（避免重复起批）
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            too_close = False
            for track in self.tracks.values():
                if track['points'] and track['state'] != TrackState.TERMINATED:
                    last_p = track['points'][-1]
                    dist = np.sqrt((x - last_p.x)**2 + (y - last_p.y)**2)
                    if dist < 100:  # 100米内不重复起批
                        too_close = True
                        break
                        
            if too_close:
                continue
                
            # 创建新航迹
            track_id = self.next_track_id
            self.next_track_id += 1
            
            vr = det.get('v_out', 0)
            if r > 0:
                vx = vr * x / r
                vy = vr * y / r
                vz = 0
            else:
                vx = vy = vz = 0
                
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
            
    def _update_unassociated_tracks(self, current_time: float):
        """更新未关联的航迹"""
        for track_id, track in self.tracks.items():
            if track.get('is_associated', False):
                track['is_associated'] = False
                continue
                
            if track['state'] == TrackState.TERMINATED:
                continue
                
            track['consecutive_lost'] += 1
            
            # 外推（仅用于已确认的航迹）
            if track['prediction'] is not None and track['state'] == TrackState.CONFIRMED:
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