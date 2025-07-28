#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能航迹起批 - 基于多特征综合评分
不依赖硬门限，而是综合考虑多个因素
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
from collections import deque


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


@dataclass
class MotionPattern:
    """运动模式特征"""
    speed: float                    # 速度
    heading: float                  # 航向（度）
    acceleration: float             # 加速度
    turn_rate: float               # 转弯率
    altitude_rate: float           # 高度变化率
    motion_type: str = "unknown"   # 运动类型：straight/turning/accelerating/hovering


class IntelligentTrackInitiation:
    """智能航迹起批 - 基于多特征评分"""
    
    # 常量定义
    RAD2DEG = 180.0 / np.pi
    DEG2RAD = np.pi / 180.0
    EPS = 1e-6
    
    def __init__(self):
        # 起批参数 - 可配置
        self.prepare_times = 3      # 预备起批所需次数
        self.confirm_times = 5      # 确认起批所需次数
        self.max_lost_times = 2     # 最大连续丢失次数
        
        # 新航迹质量门限
        self.min_new_track_quality = 0.6    # 新航迹最低质量要求
        self.min_new_track_snr = 10.0       # 新航迹最低SNR要求
        
        # 关联门限
        self.min_association_score = 0.6    # 最低关联分数（更严格）
        self.strict_first_points = True     # 对前几个点更严格
        
        # 评分权重
        self.weights = {
            'position': 0.3,        # 位置匹配度
            'velocity': 0.2,        # 速度一致性
            'motion_pattern': 0.2,  # 运动模式相似度
            'snr_consistency': 0.1, # 信号强度一致性
            'time_continuity': 0.1, # 时间连续性
            'prediction_accuracy': 0.1  # 历史预测准确度
        }
        
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
            
        # 重置索引
        detections = detections.reset_index(drop=True)
        
        # 1. 更新所有航迹的运动模式
        self._update_motion_patterns()
        
        # 2. 生成智能预测
        self._generate_intelligent_predictions(current_time)
        
        # 3. 智能关联
        associations = self._intelligent_association(detections, current_time)
        
        # 4. 更新关联的航迹
        self._update_associated_tracks(associations, detections, current_time)
        
        # 5. 更新未关联的航迹
        self._update_unassociated_tracks(current_time)
        
        # 6. 智能创建新航迹
        self._intelligent_new_tracks(detections, associations, current_time)
        
        # 7. 更新航迹状态
        self._update_track_states()
        
        # 8. 删除终止的航迹
        self._remove_terminated_tracks()
        
        # 9. 更新统计
        self._update_statistics()
        
        return self.tracks
        
    def _update_motion_patterns(self):
        """更新所有航迹的运动模式"""
        for track_id, track in self.tracks.items():
            if len(track['points']) < 2:
                continue
                
            # 获取最近的点
            recent_points = track['points'][-5:]  # 最多使用最近5个点
            
            if len(recent_points) >= 2:
                # 计算运动特征
                pattern = self._calculate_motion_pattern(recent_points)
                track['motion_pattern'] = pattern
                
                # 判断运动类型
                if pattern.turn_rate < 2.0:  # 度/秒
                    if pattern.acceleration < 2.0:  # 米/秒²
                        if pattern.speed < 5.0:  # 米/秒
                            pattern.motion_type = "hovering"
                        else:
                            pattern.motion_type = "straight"
                    else:
                        pattern.motion_type = "accelerating"
                else:
                    pattern.motion_type = "turning"
                    
    def _calculate_motion_pattern(self, points: List[TrackPoint]) -> MotionPattern:
        """计算运动模式特征"""
        pattern = MotionPattern(
            speed=0, heading=0, acceleration=0, 
            turn_rate=0, altitude_rate=0
        )
        
        if len(points) < 2:
            return pattern
            
        # 计算平均速度
        speeds = []
        headings = []
        accelerations = []
        
        for i in range(len(points)):
            p = points[i]
            speed = np.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
            speeds.append(speed)
            
            if speed > self.EPS:
                heading = np.arctan2(p.vy, p.vx) * self.RAD2DEG
                headings.append(heading)
                
        pattern.speed = np.mean(speeds) if speeds else 0
        
        # 计算加速度
        if len(speeds) >= 2:
            for i in range(1, len(speeds)):
                dt = points[i].time - points[i-1].time
                if 0 < dt < 5.0:
                    acc = (speeds[i] - speeds[i-1]) / dt
                    accelerations.append(acc)
                    
        pattern.acceleration = np.mean(np.abs(accelerations)) if accelerations else 0
        
        # 计算转弯率
        if len(headings) >= 2:
            turn_rates = []
            for i in range(1, len(headings)):
                dt = points[i].time - points[i-1].time
                if 0 < dt < 5.0:
                    dh = headings[i] - headings[i-1]
                    # 处理角度环绕
                    if dh > 180:
                        dh -= 360
                    elif dh < -180:
                        dh += 360
                    turn_rates.append(abs(dh) / dt)
                    
            pattern.turn_rate = np.mean(turn_rates) if turn_rates else 0
            
        # 计算高度变化率
        if len(points) >= 2:
            altitude_rates = []
            for i in range(1, len(points)):
                dt = points[i].time - points[i-1].time
                if 0 < dt < 5.0:
                    dz = points[i].z - points[i-1].z
                    altitude_rates.append(abs(dz) / dt)
                    
            pattern.altitude_rate = np.mean(altitude_rates) if altitude_rates else 0
            
        return pattern
        
    def _generate_intelligent_predictions(self, current_time: float):
        """生成智能预测"""
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if not track['points']:
                continue
                
            # 获取运动模式
            pattern = track.get('motion_pattern')
            
            # 根据运动模式选择预测方法
            if pattern and pattern.motion_type == "turning":
                pred = self._predict_turning(track, current_time)
            elif pattern and pattern.motion_type == "accelerating":
                pred = self._predict_accelerating(track, current_time)
            else:
                pred = self._predict_straight(track, current_time)
                
            track['prediction'] = pred
            
    def _predict_straight(self, track: Dict, current_time: float) -> TrackPoint:
        """直线运动预测"""
        last_point = track['points'][-1]
        dt = current_time - last_point.time
        
        if dt <= 0 or dt > 20.0:
            return None
            
        # 使用平均速度
        if len(track['points']) >= 3:
            # 计算最近几个点的平均速度
            vx_list = [p.vx for p in track['points'][-3:]]
            vy_list = [p.vy for p in track['points'][-3:]]
            vz_list = [p.vz for p in track['points'][-3:]]
            
            avg_vx = np.mean(vx_list)
            avg_vy = np.mean(vy_list)
            avg_vz = np.mean(vz_list)
        else:
            avg_vx = last_point.vx
            avg_vy = last_point.vy
            avg_vz = last_point.vz
            
        pred = TrackPoint(
            x = last_point.x + avg_vx * dt,
            y = last_point.y + avg_vy * dt,
            z = max(0, last_point.z + avg_vz * dt),
            vx = avg_vx,
            vy = avg_vy,
            vz = avg_vz,
            range = 0,
            azim_deg = 0,
            elev_deg = 0,
            vr = 0,
            time = current_time,
            snr = last_point.snr
        )
        
        # 更新极坐标
        self._update_polar_coords(pred)
        return pred
        
    def _predict_turning(self, track: Dict, current_time: float) -> TrackPoint:
        """转弯运动预测"""
        if len(track['points']) < 3:
            return self._predict_straight(track, current_time)
            
        last_point = track['points'][-1]
        dt = current_time - last_point.time
        
        if dt <= 0 or dt > 20.0:
            return None
            
        # 估计转弯率
        pattern = track.get('motion_pattern')
        if pattern:
            omega = pattern.turn_rate * self.DEG2RAD  # 转换为弧度/秒
        else:
            omega = 0
            
        # 使用简化的转弯模型
        v = np.sqrt(last_point.vx**2 + last_point.vy**2)
        heading = np.arctan2(last_point.vy, last_point.vx)
        
        new_heading = heading + omega * dt
        
        pred = TrackPoint(
            x = last_point.x + v * dt * np.cos(new_heading),
            y = last_point.y + v * dt * np.sin(new_heading),
            z = max(0, last_point.z + last_point.vz * dt),
            vx = v * np.cos(new_heading),
            vy = v * np.sin(new_heading),
            vz = last_point.vz,
            range = 0,
            azim_deg = 0,
            elev_deg = 0,
            vr = 0,
            time = current_time,
            snr = last_point.snr
        )
        
        self._update_polar_coords(pred)
        return pred
        
    def _predict_accelerating(self, track: Dict, current_time: float) -> TrackPoint:
        """加速运动预测"""
        if len(track['points']) < 2:
            return self._predict_straight(track, current_time)
            
        last_point = track['points'][-1]
        dt = current_time - last_point.time
        
        if dt <= 0 or dt > 20.0:
            return None
            
        # 估计加速度
        pattern = track.get('motion_pattern')
        if pattern and pattern.acceleration > 0:
            # 根据历史加速度模式预测
            acc_factor = 1.0 + pattern.acceleration * dt / pattern.speed
            acc_factor = np.clip(acc_factor, 0.5, 2.0)  # 限制加速度影响
        else:
            acc_factor = 1.0
            
        pred = TrackPoint(
            x = last_point.x + last_point.vx * dt * acc_factor,
            y = last_point.y + last_point.vy * dt * acc_factor,
            z = max(0, last_point.z + last_point.vz * dt),
            vx = last_point.vx * acc_factor,
            vy = last_point.vy * acc_factor,
            vz = last_point.vz,
            range = 0,
            azim_deg = 0,
            elev_deg = 0,
            vr = 0,
            time = current_time,
            snr = last_point.snr
        )
        
        self._update_polar_coords(pred)
        return pred
        
    def _update_polar_coords(self, point: TrackPoint):
        """更新极坐标"""
        point.range = np.sqrt(point.x**2 + point.y**2 + point.z**2)
        point.azim_deg = np.arctan2(point.x, -point.y) * self.RAD2DEG
        if point.azim_deg < 0:
            point.azim_deg += 360
            
        if point.range > self.EPS:
            point.elev_deg = np.arcsin(point.z / point.range) * self.RAD2DEG
            point.vr = -(point.vx * point.x + point.vy * point.y + point.vz * point.z) / point.range
            
    def _intelligent_association(self, detections: pd.DataFrame, 
                               current_time: float) -> Dict[int, int]:
        """智能关联 - 基于多特征评分"""
        associations = {}
        used_detections = set()
        
        # 计算每个检测点的特征
        det_features = []
        for idx, det in detections.iterrows():
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            vr = det.get('v_out', 0)
            snr = det.get('SNR/10', 10)
            
            det_features.append({
                'idx': idx,
                'x': x, 'y': y, 'z': z,
                'vr': vr,
                'snr': snr,
                'det': det
            })
            
        # 对每条航迹寻找最佳匹配
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            if track.get('prediction') is None:
                continue
                
            best_score = 0  # 改为最大化分数
            best_idx = -1
            
            for det_feat in det_features:
                if det_feat['idx'] in used_detections:
                    continue
                    
                # 计算综合评分
                score = self._calculate_association_score(track, det_feat, current_time)
                
                # 对初始点更严格的阈值
                min_score = self.min_association_score
                if self.strict_first_points and len(track['points']) < 3:
                    min_score = 0.7  # 前3个点要求更高的匹配度
                
                if score > best_score and score > min_score:
                    best_score = score
                    best_idx = det_feat['idx']
                    
            # 关联最佳匹配
            if best_idx >= 0:
                associations[track_id] = best_idx
                used_detections.add(best_idx)
                print(f"航迹 T{track_id} 关联到检测点 {best_idx}, 评分: {best_score:.2f}")
                
        return associations
        
    def _calculate_association_score(self, track: Dict, det_feat: Dict, 
                                   current_time: float) -> float:
        """计算关联评分（0-1）"""
        scores = {}
        pred = track['prediction']
        
        # 1. 位置匹配度
        dr = np.sqrt((det_feat['x'] - pred.x)**2 + 
                    (det_feat['y'] - pred.y)**2 + 
                    (det_feat['z'] - pred.z)**2)
        
        # 根据目标速度动态调整距离评分
        pattern = track.get('motion_pattern')
        if pattern:
            expected_error = pattern.speed * 2.0  # 速度越快，允许误差越大
            expected_error = max(50.0, expected_error)  # 最小50米
        else:
            expected_error = 100.0
            
        scores['position'] = np.exp(-dr / expected_error)
        
        # 2. 速度一致性
        if pred.range > self.EPS:
            dv = abs(det_feat['vr'] - pred.vr)
            expected_v_error = 10.0  # 米/秒
            scores['velocity'] = np.exp(-dv / expected_v_error)
        else:
            scores['velocity'] = 0.5
            
        # 3. 运动模式相似度
        if pattern:
            # 检查检测点是否符合预期的运动模式
            if pattern.motion_type == "straight":
                # 直线运动应该速度稳定
                scores['motion_pattern'] = 0.8
            elif pattern.motion_type == "turning":
                # 转弯运动允许更大的位置偏差
                scores['motion_pattern'] = 0.7
            elif pattern.motion_type == "hovering":
                # 悬停应该位置变化很小
                if dr < 50:
                    scores['motion_pattern'] = 0.9
                else:
                    scores['motion_pattern'] = 0.3
            else:
                scores['motion_pattern'] = 0.6
        else:
            scores['motion_pattern'] = 0.5
            
        # 4. 信号强度一致性
        if len(track['points']) >= 2:
            avg_snr = np.mean([p.snr for p in track['points'][-3:]])
            snr_diff = abs(det_feat['snr'] - avg_snr)
            scores['snr_consistency'] = np.exp(-snr_diff / 10.0)
        else:
            scores['snr_consistency'] = 0.5
            
        # 5. 时间连续性
        dt = current_time - track['last_update_time']
        if dt < 2.0:
            scores['time_continuity'] = 1.0
        elif dt < 5.0:
            scores['time_continuity'] = 0.8
        else:
            scores['time_continuity'] = 0.5
            
        # 6. 历史预测准确度
        accuracy = track.get('prediction_accuracy', 0.5)
        scores['prediction_accuracy'] = accuracy
        
        # 计算加权总分
        total_score = 0
        for key, weight in self.weights.items():
            total_score += weight * scores.get(key, 0)
            
        return total_score
        
    def _update_associated_tracks(self, associations: Dict[int, int], 
                                 detections: pd.DataFrame, current_time: float):
        """更新关联的航迹"""
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            det = detections.iloc[det_idx]
            
            # 转换坐标
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            # 更新预测准确度
            if track.get('prediction'):
                pred = track['prediction']
                error = np.sqrt((x - pred.x)**2 + (y - pred.y)**2)
                
                # 使用指数移动平均更新准确度
                old_accuracy = track.get('prediction_accuracy', 0.5)
                if error < 50:
                    new_accuracy = 0.9
                elif error < 100:
                    new_accuracy = 0.7
                elif error < 200:
                    new_accuracy = 0.5
                else:
                    new_accuracy = 0.3
                    
                track['prediction_accuracy'] = 0.7 * old_accuracy + 0.3 * new_accuracy
                
            # 计算速度（使用平滑）
            if len(track['points']) >= 1:
                last_point = track['points'][-1]
                dt = current_time - last_point.time
                
                if 0 < dt < 2.0:
                    # 瞬时速度
                    inst_vx = (x - last_point.x) / dt
                    inst_vy = (y - last_point.y) / dt
                    inst_vz = (z - last_point.z) / dt
                    
                    # 平滑（考虑运动模式）
                    pattern = track.get('motion_pattern')
                    if pattern and pattern.motion_type == "straight":
                        # 直线运动，速度变化应该较小
                        alpha = 0.3
                    else:
                        # 其他运动，允许速度变化更大
                        alpha = 0.5
                        
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
            if len(track['points']) > 30:  # 减少历史长度
                track['points'] = track['points'][-30:]
                
    def _intelligent_new_tracks(self, detections: pd.DataFrame, 
                               associations: Dict[int, int], current_time: float):
        """智能创建新航迹"""
        used_indices = set(associations.values())
        
        # 对未关联的检测点进行聚类或质量评估
        unassociated_dets = []
        
        for idx, det in detections.iterrows():
            if idx in used_indices:
                continue
                
            # 基本质量过滤
            snr = det.get('SNR/10', 0)
            if snr < self.min_new_track_snr:
                continue
                
            # 坐标转换
            r = det['range_out']
            azim_rad = det['azim_out'] * self.DEG2RAD
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)
            z = det.get('high', 0)
            
            unassociated_dets.append({
                'idx': idx,
                'x': x, 'y': y, 'z': z,
                'det': det,
                'quality': self._evaluate_new_track_quality(det)
            })
            
        # 只为高质量的检测点创建新航迹
        for det_info in unassociated_dets:
            if det_info['quality'] < self.min_new_track_quality:
                continue
                
            # 检查是否离现有航迹太近（避免重复）
            too_close = False
            for track in self.tracks.values():
                if track['state'] == TrackState.TERMINATED:
                    continue
                    
                if track.get('prediction'):
                    pred = track['prediction']
                    dist = np.sqrt((det_info['x'] - pred.x)**2 + 
                                 (det_info['y'] - pred.y)**2)
                    
                    # 动态距离阈值
                    pattern = track.get('motion_pattern')
                    if pattern:
                        min_dist = max(30, pattern.speed * 3)  # 3秒的运动距离
                    else:
                        min_dist = 50
                        
                    if dist < min_dist:
                        too_close = True
                        break
                        
            if too_close:
                continue
                
            # 创建新航迹
            self._create_new_track(det_info, current_time)
            
    def _evaluate_new_track_quality(self, det: pd.Series) -> float:
        """评估新航迹质量"""
        quality = 0.0
        
        # SNR贡献
        snr = det.get('SNR/10', 0)
        if snr > 15:
            quality += 0.4
        elif snr > 10:
            quality += 0.3
        elif snr > 8:
            quality += 0.2
        else:
            quality += 0.1
            
        # 速度合理性
        vr = abs(det.get('v_out', 0))
        if 5 < vr < 300:  # 合理的速度范围（米/秒）
            quality += 0.3
        elif vr < 5:  # 慢速目标
            quality += 0.2
        else:
            quality += 0.1
            
        # 距离合理性
        r = det['range_out']
        if 500 < r < 50000:  # 合理的距离范围（米）
            quality += 0.3
        else:
            quality += 0.2
            
        return quality
        
    def _create_new_track(self, det_info: Dict, current_time: float):
        """创建新航迹"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        det = det_info['det']
        x = det_info['x']
        y = det_info['y'] 
        z = det_info['z']
        
        # 估计初始速度
        r = det['range_out']
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
            'motion_pattern': None,
            'prediction_accuracy': 0.5  # 初始准确度
        }
        
        self.stats['total_created'] += 1
        print(f"创建新航迹 T{track_id} (质量: {det_info['quality']:.2f})")
        
    def _update_track_states(self):
        """更新航迹状态"""
        for track in self.tracks.values():
            if track['consecutive_lost'] > self.max_lost_times:
                track['state'] = TrackState.TERMINATED
                continue
                
            track_times = track['track_times']
            
            # 考虑预测准确度
            accuracy = track.get('prediction_accuracy', 0.5)
            
            # 高准确度的航迹更快确认
            if accuracy > 0.7:
                confirm_threshold = self.confirm_times - 1
            else:
                confirm_threshold = self.confirm_times
                
            if track_times >= confirm_threshold:
                if track['state'] != TrackState.CONFIRMED:
                    print(f"航迹 T{track['track_id']} 确认起批 "
                          f"(点数: {len(track['points'])}, 准确度: {accuracy:.2f})")
                track['state'] = TrackState.CONFIRMED
            elif track_times >= self.prepare_times:
                if track['state'] == TrackState.TENTATIVE:
                    print(f"航迹 T{track['track_id']} 预备起批")
                track['state'] = TrackState.PREPARE
                
    def _update_unassociated_tracks(self, current_time: float):
        """更新未关联的航迹"""
        for track_id, track in self.tracks.items():
            if track.get('is_associated', False):
                track['is_associated'] = False
                continue
                
            if track['state'] == TrackState.TERMINATED:
                continue
                
            track['consecutive_lost'] += 1
            
            # 只对高准确度的确认航迹进行外推
            if (track['state'] == TrackState.CONFIRMED and 
                track.get('prediction_accuracy', 0) > 0.6 and
                track.get('prediction')):
                
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
            
            # 添加运动模式信息
            pattern = track.get('motion_pattern')
            motion_info = ""
            if pattern:
                motion_info = f"{pattern.motion_type} (速度:{pattern.speed:.1f}m/s)"
                
            display_tracks[track_id] = {
                'points': points,
                'established_mode': mode_map.get(track['state'], 0),
                'track_times': track['track_times'],
                'consecutive_lost': track['consecutive_lost'],
                'prediction_accuracy': track.get('prediction_accuracy', 0),
                'motion_info': motion_info
            }
            
        return display_tracks