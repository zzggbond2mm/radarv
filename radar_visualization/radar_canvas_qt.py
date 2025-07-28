from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsPathItem, QApplication, QGraphicsObject, QGraphicsItemGroup, QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QTimer, QPropertyAnimation, pyqtProperty, QEasingCurve
from PyQt5.QtGui import QColor, QPen, QBrush, QTransform, QFont, QPainterPath, QPainter, QPolygonF
from PyQt5.QtCore import QPointF
import numpy as np
import pandas as pd
import math

class PointItem(QGraphicsObject):
    """一个代表单个数据点的自定义图形项，可以处理自己的点击事件。"""
    point_clicked = pyqtSignal(object)  # 定义一个信号，当被点击时发射自身

    def __init__(self, x, y, size, color, data_dict):
        super().__init__()
        self.setPos(x, y)
        self._size = size
        self._base_size = size  # 保存基础大小
        self._color = color
        self._scale_factor = 1.0  # 动画缩放因子
        
        # 将原始数据存储在item中
        self.point_data = data_dict
        self.setData(0, data_dict)
        
        # 启用悬停事件，改变鼠标样式
        self.setAcceptHoverEvents(True)
        
        # 为高概率XGBoost点添加呼吸动画
        self._setup_animation()
    
    def _setup_animation(self):
        """设置动画效果"""
        # 检查是否是高概率XGBoost点
        if ('xgb_probability' in self.point_data and 
            self.point_data['xgb_probability'] is not None and 
            self.point_data['xgb_probability'] > 0.7):
            
            # 创建缩放动画
            self.animation = QPropertyAnimation(self, b"scaleFactor")
            self.animation.setDuration(2000)  # 2秒周期
            self.animation.setStartValue(0.9)
            self.animation.setEndValue(1.1)
            self.animation.setLoopCount(-1)  # 无限循环
            
            # 设置动画曲线
            self.animation.setEasingCurve(QEasingCurve.InOutSine)
            
            # 启动动画
            self.animation.start()
        else:
            self.animation = None
    
    @pyqtProperty(float)
    def scaleFactor(self):
        return self._scale_factor
    
    @scaleFactor.setter
    def scaleFactor(self, value):
        self._scale_factor = value
        self.update()  # 触发重绘
    
    def boundingRect(self):
        """返回项的边界矩形"""
        animated_size = self._size * self._scale_factor
        return QRectF(-self._size / 2, -self._size / 2, self._size, self._size)

    def paint(self, painter, option, widget):
        """绘制项 - 美化版本"""
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        
        # 根据数据类型选择不同的绘制风格
        point_data = self.point_data
        
        if 'xgb_probability' in point_data and point_data['xgb_probability'] is not None:
            # XGBoost预测结果 - 使用特殊的美化效果
            self._paint_xgboost_point(painter, point_data['xgb_probability'])
        elif 'confidence' in point_data and point_data['confidence'] is not None:
            # LSTM预测结果 - 使用钻石形状
            self._paint_lstm_point(painter, point_data['confidence'])
        else:
            # 普通点 - 简单圆形
            self._paint_normal_point(painter, point_data.get('SNR/10', 10))
    
    def _paint_xgboost_point(self, painter, probability):
        """绘制XGBoost预测点 - 美化版"""
        rect = self.boundingRect()
        
        # 根据概率调整颜色和效果
        if probability > 0.8:
            # 高概率 - 深蓝色渐变 + 发光效果
            base_color = QColor(0, 80, 255)  # 鲜艳深蓝色
            glow_color = QColor(100, 180, 255, 120)  # 亮蓝色光晕
            border_color = QColor(255, 255, 255, 220)
            alpha = 250
        elif probability > 0.6:
            # 中等概率 - 蓝色
            base_color = QColor(50, 150, 255)  # 明亮蓝色
            glow_color = QColor(120, 200, 255, 100)
            border_color = QColor(255, 255, 255, 180)
            alpha = 210
        else:
            # 低概率 - 浅蓝色
            base_color = QColor(120, 200, 255)  # 浅蓝色
            glow_color = QColor(160, 220, 255, 80)
            border_color = QColor(255, 255, 255, 160)
            alpha = 170
        
        # 设置透明度
        base_color.setAlpha(alpha)
        
        # 绘制外层光晕（考虑动画缩放）
        glow_size = rect.width() * 1.4
        glow_rect = QRectF(-glow_size/2, -glow_size/2, glow_size, glow_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(glow_rect)
        
        # 绘制主体 - 六边形或圆形
        painter.setPen(QPen(border_color, 3))  # 更粗的边框
        painter.setBrush(QBrush(base_color))
        
        if probability > 0.7:
            # 高概率用六边形
            self._draw_hexagon(painter, rect)
        else:
            # 中低概率用圆形
            painter.drawEllipse(rect)
        
        # 在中心添加发光的小圆点
        center_size = rect.width() * 0.25
        center_rect = QRectF(-center_size/2, -center_size/2, center_size, center_size)
        
        # 高亮中心点
        center_glow = QColor(255, 255, 255, 250)
        painter.setBrush(QBrush(center_glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_rect)
    
    def _paint_lstm_point(self, painter, confidence):
        """绘制LSTM预测点 - 钻石形状"""
        rect = self.boundingRect()
        
        # LSTM用红色系，增加渐变效果
        if confidence > 0.8:
            color = QColor(255, 60, 60, 230)   # 鲜红色
            glow_color = QColor(255, 120, 120, 100)
            border_color = QColor(255, 255, 255, 200)
        elif confidence > 0.6:
            color = QColor(255, 140, 60, 210)  # 橙红色
            glow_color = QColor(255, 180, 120, 80)
            border_color = QColor(255, 255, 255, 180)
        else:
            color = QColor(255, 220, 60, 190)  # 金黄色
            glow_color = QColor(255, 240, 120, 60)
            border_color = QColor(255, 255, 255, 160)
        
        # 绘制光晕
        glow_size = rect.width() * 1.3
        glow_rect = QRectF(-glow_size/2, -glow_size/2, glow_size, glow_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(glow_rect)
        
        # 绘制钻石形状
        painter.setPen(QPen(border_color, 3))
        painter.setBrush(QBrush(color))
        self._draw_diamond(painter, rect)
        
        # 添加中心高光
        center_size = rect.width() * 0.2
        center_rect = QRectF(-center_size/2, -center_size/2, center_size, center_size)
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_rect)
    
    def _paint_normal_point(self, painter, snr):
        """绘制普通点"""
        rect = self.boundingRect()
        
        # 根据SNR调整颜色，增加透明度渐变
        if snr > 20:
            color = QColor(255, 120, 120, 200)  # 红色
            glow_color = QColor(255, 160, 160, 60)
            border_color = QColor(255, 255, 255, 180)
        elif snr > 15:
            color = QColor(255, 255, 120, 180)  # 黄色
            glow_color = QColor(255, 255, 180, 50)
            border_color = QColor(255, 255, 255, 160)
        else:
            color = QColor(120, 255, 120, 160)  # 绿色
            glow_color = QColor(160, 255, 160, 40)
            border_color = QColor(255, 255, 255, 140)
        
        # 绘制光晕
        glow_size = rect.width() * 1.2
        glow_rect = QRectF(-glow_size/2, -glow_size/2, glow_size, glow_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(glow_rect)
        
        # 绘制主体
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(border_color, 2))
        painter.drawEllipse(rect)
        
        # 添加中心点
        center_size = rect.width() * 0.15
        center_rect = QRectF(-center_size/2, -center_size/2, center_size, center_size)
        painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_rect)
    
    def _draw_hexagon(self, painter, rect):
        """绘制六边形"""
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 * 0.8  # 稍微缩小
        
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            points.append(QPointF(x, y))
        
        polygon = QPolygonF(points)
        painter.drawPolygon(polygon)
    
    def _draw_diamond(self, painter, rect):
        """绘制钻石形状"""
        center = rect.center()
        half_width = rect.width() / 2 * 0.8
        half_height = rect.height() / 2 * 0.8
        
        points = [
            QPointF(center.x(), center.y() - half_height),  # 上
            QPointF(center.x() + half_width, center.y()),   # 右
            QPointF(center.x(), center.y() + half_height),  # 下
            QPointF(center.x() - half_width, center.y())    # 左
        ]
        
        polygon = QPolygonF(points)
        painter.drawPolygon(polygon)

    def mousePressEvent(self, event):
        """鼠标点击事件，发射信号"""
        self.point_clicked.emit(self)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        """鼠标进入事件，显示手形光标"""
        QApplication.instance().setOverrideCursor(Qt.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """鼠标离开事件，恢复默认光标"""
        QApplication.instance().restoreOverrideCursor()
        super().hoverLeaveEvent(event)


class QtRadarCanvas(QGraphicsView):
    """
    一个使用QGraphicsView实现的雷达显示画布，用于替代Matplotlib。
    - 支持高性能的图形渲染。
    - 支持流畅的缩放和平移。
    """
    point_clicked = pyqtSignal(dict) # 定义一个信号，将点的信息传递给主窗口
    zoom_requested = pyqtSignal(int) # 定义缩放请求信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_range = 15000  # 最大显示范围 (米)

        # 设置场景
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor('#001100'))
        self.setScene(self.scene)

        # 设置视图属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag) # 启用手形拖动
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        
        # 启用缩放功能的关键设置
        self.setInteractive(True)
        self.setMouseTracking(True)
        
        # 设置滚轮缩放的限制
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        # 启用滚轮事件
        self.setFocusPolicy(Qt.WheelFocus)
        
        # 设置缩放范围限制
        self._min_scale = 0.01   # 降低最小缩放限制，允许更大的缩小
        self._max_scale = 50.0   # 增加最大缩放限制，允许更大的放大
        
        # 确保视图可以接收滚轮事件
        self.viewport().setFocusPolicy(Qt.WheelFocus)
        
        # 视图范围
        # 我们使用一个很大的场景矩形，然后通过fitInView来控制初始视图
        scene_rect = QRectF(-self.max_range, -self.max_range, self.max_range * 2, self.max_range * 2)
        self.scene.setSceneRect(scene_rect)
        
        # 绘制背景
        self.draw_background_grid()
        
        # 用于存储点和航迹的item
        self.point_items = []
        self.track_items = []
        self.highlight_item = None
        self._points_group = None
        self._tracks_group = None
        self._distance_circles_group = None # 新增：用于存储距离圈的组

        # 优化：定时器用于批量更新
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(16) # ~60fps
        
        # 初始化点组
        self._points_group = QGraphicsItemGroup()
        self.scene.addItem(self._points_group)
        
        # 初始化航迹组
        self._tracks_group = QGraphicsItemGroup()
        self.scene.addItem(self._tracks_group)
        
        # 新增：初始化距离圈组
        self._distance_circles_group = QGraphicsItemGroup()
        self.scene.addItem(self._distance_circles_group)

    def draw_background_grid(self):
        """绘制雷达背景网格（同心圆和方位线）"""
        # --- 绘制同心圆 ---
        pen = QPen(QColor('green'), 0) # 使用0宽度的笔，可以自适应缩放
        pen.setStyle(Qt.DotLine)
        
        # 确定距离间隔
        if self.max_range > 10000:
            step = 3000
        elif self.max_range > 5000:
            step = 2000
        else:
            step = 1000
        
        ranges = np.arange(step, self.max_range + step, step)
        
        for r in ranges:
            circle = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pen)
            self.scene.addItem(circle)
            
            # 添加距离标签
            label = QGraphicsTextItem(f'{r/1000:.0f}km')
            label.setDefaultTextColor(QColor('lightgreen'))
            label.setFont(QFont("Arial", 150))
            # 将标签放在圆的右侧
            label_rect = label.boundingRect()
            label.setPos(r - label_rect.width() / 2, -label_rect.height() / 2)
            self.scene.addItem(label)

        # --- 绘制最外圈实线 ---
        pen.setStyle(Qt.SolidLine)
        pen.setColor(QColor('#00FF00'))
        outer_circle = QGraphicsEllipseItem(-self.max_range, -self.max_range, self.max_range * 2, self.max_range * 2)
        outer_circle.setPen(pen)
        self.scene.addItem(outer_circle)
        
        # --- 绘制方位线 ---
        pen.setStyle(Qt.SolidLine)
        pen.setColor(QColor('green'))
        angles = np.arange(0, 360, 30)

        for angle_deg in angles:
            angle_rad = np.radians(angle_deg - 90) # QGraphicsView 0度在右边, 我们需要转为北向
            
            x_end = self.max_range * np.cos(angle_rad)
            y_end = self.max_range * np.sin(angle_rad)
            line = QGraphicsLineItem(0, 0, x_end, y_end)
            line.setPen(pen)
            self.scene.addItem(line)

            # 添加角度标签
            label_r = self.max_range * 1.05
            x_label = label_r * np.cos(angle_rad)
            y_label = label_r * np.sin(angle_rad)
            
            label = QGraphicsTextItem(f'{angle_deg}°')
            label.setDefaultTextColor(QColor('lightgreen'))
            label.setFont(QFont("Arial", 150))
            label_rect = label.boundingRect()
            label.setPos(x_label - label_rect.width()/2, y_label - label_rect.height()/2)
            self.scene.addItem(label)

    def wheelEvent(self, event):
        """鼠标滚轮事件，处理缩放"""
        # 确保事件有效
        if not event.angleDelta().y():
            event.ignore()
            return
            
        # 缩放因子
        zoom_in_factor = 1.2   # 稍微增加缩放灵敏度
        zoom_out_factor = 1 / zoom_in_factor
        
        # 获取滚轮方向
        angle_delta = event.angleDelta().y()
        
        # 确定缩放因子
        if angle_delta > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor
        
        # 检查缩放限制
        current_scale = self.transform().m11()  # 获取当前缩放级别
        new_scale = current_scale * scale_factor
        
        if new_scale < self._min_scale or new_scale > self._max_scale:
            print(f"缩放限制：当前{current_scale:.3f}, 新{new_scale:.3f}, 范围[{self._min_scale}, {self._max_scale}]")
            event.ignore()
            return
        
        # 获取鼠标在视图中的位置
        view_pos = event.pos()
        
        # 获取鼠标在场景中的位置（缩放前）
        scene_pos = self.mapToScene(view_pos)
        
        # 执行缩放
        self.scale(scale_factor, scale_factor)
        
        # 获取鼠标在场景中的新位置（缩放后）
        new_scene_pos = self.mapToScene(view_pos)
        
        # 计算偏移并进行补偿，实现以鼠标为中心的缩放
        delta = new_scene_pos - scene_pos
        self.translate(delta.x(), delta.y())
        
        # 打印调试信息
        print(f"缩放：因子{scale_factor:.3f}, 当前缩放{self.transform().m11():.3f}")
        
        # 发射信号以便主窗口同步其他画布（如果有的话）
        self.zoom_requested.emit(angle_delta)
        
        # 接受事件
        event.accept()

    def showEvent(self, event):
        """视图首次显示时，自动缩放到合适的大小"""
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().showEvent(event)

    def update_display(self, data, show_tracks=False, track_data=None, initiated_tracks=None):
        """
        更新显示。
        data: 一个包含要显示点的DataFrame。
        show_tracks: 是否显示航迹。
        track_data: 航迹数据。
        initiated_tracks: 航迹起批数据（新增）
        """
        self.clear_display()
        if data is None or data.empty:
            return

        for _, row in data.iterrows():
            # 1. 坐标转换
            r = row['range_out']
            # Azimuth 0 is North (up, -Y), 90 is East (right, +X)
            # x = r * sin(azi), y = -r * cos(azi)
            azim_rad = np.radians(row['azim_out'])
            x = r * np.sin(azim_rad)
            y = -r * np.cos(azim_rad)

            # 2. 根据属性确定大小（颜色在paint方法中处理）
            size = 150  # 默认大小
            color = QColor('white')  # 占位颜色，实际在paint中设置

            # 根据数据类型调整大小
            if 'xgb_probability' in row and row['xgb_probability'] is not None:
                # XGBoost预测结果 - 根据概率调整大小
                probability = row['xgb_probability']
                if probability > 0.8:
                    size = 400  # 高概率 - 大尺寸
                elif probability > 0.6:
                    size = 300  # 中等概率 - 中等尺寸
                else:
                    size = 200  # 低概率 - 小尺寸
            elif 'confidence' in row and row['confidence'] is not None:
                # LSTM预测结果
                confidence = row['confidence']
                if confidence > 0.8:
                    size = 450
                elif confidence > 0.6:
                    size = 350
                else:
                    size = 250
            else:
                # 普通数据 - 根据SNR调整大小
                snr = row.get('SNR/10', 10)
                if snr > 20:
                    size = 350
                elif snr > 15:
                    size = 250
                else:
                    size = 150

            # 3. 创建自定义的PointItem
            point = PointItem(x, y, size, color, row.to_dict())
            point.point_clicked.connect(self.on_point_item_clicked) # 连接子项的信号

            self.scene.addItem(point)
            self.point_items.append(point)
        
        # 显示原始航迹（来自文件）
        if show_tracks and track_data:
            self._draw_original_tracks(track_data)
        
        # 显示航迹起批结果（新增）
        if initiated_tracks:
            self._draw_initiated_tracks(initiated_tracks)

    def _draw_original_tracks(self, track_data):
        """绘制原始航迹（来自文件）"""
        track_pen = QPen(QColor('cyan'), 0)
        track_pen.setWidth(50) # Give tracks a visible width
        for batch_num, track_points in track_data.items():
            if len(track_points) > 1:
                track_df = pd.DataFrame(track_points)
                path = QPainterPath()
                
                # First point
                row = track_df.iloc[0]
                r = row['range_out']
                azim_rad = np.radians(row['azim_out'])
                x = r * np.sin(azim_rad)
                y = -r * np.cos(azim_rad)
                path.moveTo(x, y)
                
                # Subsequent points
                for _, row in track_df.iloc[1:].iterrows():
                    r = row['range_out']
                    azim_rad = np.radians(row['azim_out'])
                    x = r * np.sin(azim_rad)
                    y = -r * np.cos(azim_rad)
                    path.lineTo(x, y)

                track_item = QGraphicsPathItem(path)
                track_item.setPen(track_pen)
                self.scene.addItem(track_item)
                self.track_items.append(track_item)

    def _draw_initiated_tracks(self, initiated_tracks):
        """绘制航迹起批结果"""
        print(f"🎨 开始绘制起批航迹，数量: {len(initiated_tracks) if initiated_tracks else 0}")
        
        # 定义不同状态的航迹样式
        track_styles = {
            0: {'color': QColor(255, 255, 0, 150), 'width': 30, 'style': Qt.DotLine, 'name': '待定'},      # 黄色虚线 - 待定
            1: {'color': QColor(255, 165, 0, 200), 'width': 50, 'style': Qt.DashLine, 'name': '预备'},     # 橙色破折线 - 预备起批  
            2: {'color': QColor(0, 255, 0, 255), 'width': 80, 'style': Qt.SolidLine, 'name': '确认'}       # 绿色实线 - 确认起批
        }
        
        for track_id, track_info in initiated_tracks.items():
            track_points = track_info['points']
            established_mode = track_info['established_mode']
            
            if len(track_points) < 1:  # 改为至少需要1个点
                continue
                
            # 获取对应状态的样式
            style = track_styles.get(established_mode, track_styles[0])
            
            # 创建画笔
            track_pen = QPen(style['color'], 0)
            track_pen.setWidth(style['width'])
            track_pen.setStyle(style['style'])
            
            if len(track_points) == 1:
                # 单点航迹，显示为圆形标记
                point = track_points[0]
                marker_size = style['width'] * 3  # 根据航迹宽度调整标记大小
                marker = QGraphicsEllipseItem(-marker_size/2, -marker_size/2, marker_size, marker_size)
                marker.setPos(point['x'], point['y'])
                
                # 设置标记样式
                marker.setPen(track_pen)
                marker.setBrush(QBrush(style['color']))
                
                self.scene.addItem(marker)
                self.track_items.append(marker)
                print(f"  ✅ 已添加单点航迹 T{track_id} ({style['name']})，标记大小: {marker_size}")
                
                # 在标记旁边添加标签
                self._add_track_label(
                    point['x'], 
                    point['y'], 
                    f"T{track_id}({style['name']})", 
                    style['color']
                )
            else:
                # 多点航迹，绘制路径
                path = QPainterPath()
                
                # 第一个点
                first_point = track_points[0]
                path.moveTo(first_point['x'], first_point['y'])
                
                # 后续点
                for point in track_points[1:]:
                    path.lineTo(point['x'], point['y'])
                
                # 创建航迹项
                track_item = QGraphicsPathItem(path)
                track_item.setPen(track_pen)
                self.scene.addItem(track_item)
                self.track_items.append(track_item)
                print(f"  ✅ 已添加航迹 T{track_id} ({style['name']})，点数: {len(track_points)}")
                
                # 在航迹末端添加状态标签
                if track_points:
                    last_point = track_points[-1]
                    self._add_track_label(
                        last_point['x'], 
                        last_point['y'], 
                        f"T{track_id}({style['name']})", 
                        style['color']
                    )
                    
                # 在航迹起点添加起始标记
                if track_points:
                    first_point = track_points[0]
                    self._add_track_start_marker(
                        first_point['x'], 
                        first_point['y'], 
                        style['color']
                    )

    def _add_track_label(self, x, y, text, color):
        """在指定位置添加航迹标签"""
        label = QGraphicsTextItem(text)
        label.setDefaultTextColor(color)
        label.setFont(QFont("Arial", 120, QFont.Bold))  # 设置字体
        
        # 调整标签位置，使其不与航迹重叠
        label_rect = label.boundingRect()
        label.setPos(x + 200, y - label_rect.height() / 2)
        
        # 添加背景框以提高可读性
        background = QGraphicsRectItem(label.boundingRect())
        background.setBrush(QBrush(QColor(0, 0, 0, 128)))  # 半透明黑色背景
        background.setPen(QPen(Qt.NoPen))
        background.setParentItem(label)
        
        self.scene.addItem(label)
        self.track_items.append(label)

    def _add_track_start_marker(self, x, y, color):
        """在航迹起点添加标记"""
        # 创建一个小圆圈标记起点
        marker_size = 300
        marker = QGraphicsEllipseItem(-marker_size/2, -marker_size/2, marker_size, marker_size)
        marker.setPos(x, y)
        
        # 设置标记样式
        pen = QPen(color, 80)
        pen.setStyle(Qt.SolidLine)
        marker.setPen(pen)
        marker.setBrush(QBrush(color))
        
        self.scene.addItem(marker)
        self.track_items.append(marker)

    def clear_display(self):
        """清除所有数据点，但保留航迹"""
        for item in self.point_items:
            self.scene.removeItem(item)
        
        self.point_items.clear()
        
        if self.highlight_item:
            self.scene.removeItem(self.highlight_item)
            self.highlight_item = None

    def clear_all_display(self):
        """清除所有显示内容，包括航迹"""
        for item in self.point_items:
            self.scene.removeItem(item)
        for item in self.track_items:
            self.scene.removeItem(item)
        
        self.point_items.clear()
        self.track_items.clear()
        
        if self.highlight_item:
            self.scene.removeItem(self.highlight_item)
            self.highlight_item = None

    def highlight_point(self, point_item):
        """高亮选中的点"""
        # 先清除旧的高亮
        if self.highlight_item:
            self.scene.removeItem(self.highlight_item)
            self.highlight_item = None
        
        if point_item is None:
            return

        # 创建一个黄色的空心圆作为高亮标记
        size = point_item.boundingRect().width() * 1.5  # 高亮框比点稍大
        pen = QPen(QColor('yellow'), 0)
        pen.setWidth(100) # 设置一个可见的宽度
        
        highlighter = QGraphicsEllipseItem(-size/2, -size/2, size, size)
        highlighter.setPos(point_item.pos()) # 位置与被点击的点相同
        highlighter.setPen(pen)
        highlighter.setBrush(QBrush(Qt.NoBrush)) # 设置为无填充
        
        self.scene.addItem(highlighter)
        self.highlight_item = highlighter

    def on_point_item_clicked(self, point_item):
        """处理来自PointItem的点击信号"""
        self.highlight_point(point_item)
        # 发射一个带数据字典的信号给主窗口
        self.point_clicked.emit(point_item.point_data) 

    def set_distance_circles(self, visible, max_range_m, interval_m=2000):
        """
        绘制或清除距离圈和标签
        :param visible: 是否显示
        :param max_range_m: 最大显示距离（米）
        :param interval_m: 圈之间的间隔（米）
        """
        # 先清空旧的
        for item in self._distance_circles_group.childItems():
            self.scene.removeItem(item)
            del item

        if not visible:
            return

        pen = QPen(QColor("#606060"), 2, Qt.SolidLine)  # 深灰色，稍粗
        font = QFont("Arial", 40)
        font.setBold(True)
        text_color = QBrush(QColor("white"))

        num_circles = int(max_range_m / interval_m)

        for i in range(1, num_circles + 1):
            radius = i * interval_m
            
            # 绘制圆圈
            circle = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
            circle.setPen(pen)
            self._distance_circles_group.addToGroup(circle)
            
            # 在X轴正半轴添加距离标签
            label_text = f"{radius / 1000:.0f} km"
            text_item = QGraphicsTextItem(label_text)
            text_item.setFont(font)
            text_item.setDefaultTextColor(QColor("white"))
            
            # 将标签放在圆圈的右侧
            text_rect = text_item.boundingRect()
            text_item.setPos(radius - text_rect.width() / 2, -text_rect.height() / 2)
            
            self._distance_circles_group.addToGroup(text_item)
            
    def showEvent(self, event):
        """视图首次显示时，自动缩放到合适大小"""
        # 确保只在首次显示时执行一次
        if not hasattr(self, '_initial_fit_done'):
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._initial_fit_done = True
        super().showEvent(event)