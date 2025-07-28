from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsPathItem, QApplication, QGraphicsObject
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPropertyAnimation, pyqtProperty, QEasingCurve, QPointF
from PyQt5.QtGui import QColor, QPen, QBrush, QFont, QPainterPath, QPainter
import numpy as np
import math


class PointItem(QGraphicsObject):
    """雷达点图形项"""
    point_clicked = pyqtSignal(object)
    
    def __init__(self, x, y, size, data_dict):
        super().__init__()
        self.setPos(x, y)
        self._size = size
        self.point_data = data_dict
        self.setData(0, data_dict)
        self.setAcceptHoverEvents(True)
        self._scale_factor = 1.0
        self._animation = None
        
        # 高置信度点添加动画
        if data_dict.get('xgb_probability', 0) > 0.8:
            self._setup_animation()
            
    def _setup_animation(self):
        """设置呼吸动画"""
        self._animation = QPropertyAnimation(self, b"scaleFactor")
        self._animation.setDuration(2000)
        self._animation.setStartValue(0.9)
        self._animation.setEndValue(1.1)
        self._animation.setLoopCount(-1)
        self._animation.setEasingCurve(QEasingCurve.InOutSine)
        self._animation.start()
        
    @pyqtProperty(float)
    def scaleFactor(self):
        return self._scale_factor
        
    @scaleFactor.setter
    def scaleFactor(self, value):
        self._scale_factor = value
        self.update()
        
    def boundingRect(self):
        s = self._size * self._scale_factor
        return QRectF(-s/2, -s/2, s, s)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 根据数据类型绘制
        if 'xgb_probability' in self.point_data:
            self._paint_xgb_point(painter)
        else:
            self._paint_normal_point(painter)
            
    def _paint_xgb_point(self, painter):
        """绘制XGBoost点"""
        prob = self.point_data['xgb_probability']
        
        # 根据概率选择颜色
        if prob > 0.8:
            color = QColor(0, 100, 255, 220)
            glow_color = QColor(0, 150, 255, 80)
        elif prob > 0.6:
            color = QColor(50, 150, 255, 200)
            glow_color = QColor(100, 200, 255, 60)
        else:
            color = QColor(120, 200, 255, 180)
            glow_color = QColor(150, 220, 255, 40)
            
        # 绘制光晕
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow_color))
        glow_rect = self.boundingRect().adjusted(-5, -5, 5, 5)
        painter.drawEllipse(glow_rect)
        
        # 绘制主体
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self.boundingRect())
        
    def _paint_normal_point(self, painter):
        """绘制普通点"""
        snr = self.point_data.get('SNR/10', 10)
        
        if snr > 20:
            color = QColor(255, 100, 100, 200)
        elif snr > 15:
            color = QColor(255, 200, 100, 180)
        else:
            color = QColor(100, 255, 100, 160)
            
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self.boundingRect())
        
    def mousePressEvent(self, event):
        self.point_clicked.emit(self)
        super().mousePressEvent(event)
        
    def hoverEnterEvent(self, event):
        QApplication.instance().setOverrideCursor(Qt.PointingHandCursor)
        
    def hoverLeaveEvent(self, event):
        QApplication.instance().restoreOverrideCursor()


class QtRadarCanvasV2(QGraphicsView):
    """改进的雷达画布V2"""
    point_clicked = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.max_range = 15000  # 最大范围（米）
        
        # 创建场景
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor(0, 10, 0))  # 深绿色背景
        self.setScene(self.scene)
        
        # 设置视图属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 设置场景矩形
        scene_size = self.max_range + 2000  # 留出边距
        self.scene.setSceneRect(-scene_size, -scene_size, scene_size * 2, scene_size * 2)
        
        # 绘制背景网格
        self._draw_grid()
        
        # 存储显示项
        self.point_items = []
        self.track_items = []
        self.highlight_item = None
        
    def _draw_grid(self):
        """绘制雷达网格"""
        # 绘制同心圆
        for r in range(3000, int(self.max_range) + 1, 3000):
            # 使用不同样式
            if r == self.max_range:
                pen = QPen(QColor(0, 255, 0), 0)  # 最外圈实线
            else:
                pen = QPen(QColor(0, 150, 0), 0)  # 内圈虚线
                pen.setStyle(Qt.DotLine)
                
            circle = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pen)
            self.scene.addItem(circle)
            
            # 添加距离标签
            label = QGraphicsTextItem(f'{r/1000:.0f}km')
            label.setDefaultTextColor(QColor(150, 255, 150))
            label.setFont(QFont("Arial", 150))
            
            # 放置在右侧
            bounds = label.boundingRect()
            label.setPos(r - bounds.width()/2, -bounds.height()/2)
            self.scene.addItem(label)
            
        # 绘制方位线
        for angle in range(0, 360, 30):
            rad = np.radians(angle - 90)  # 转换为数学角度
            x_end = self.max_range * np.cos(rad)
            y_end = self.max_range * np.sin(rad)
            
            pen = QPen(QColor(0, 150, 0), 0)
            pen.setStyle(Qt.DotLine)
            
            line = QGraphicsLineItem(0, 0, x_end, y_end)
            line.setPen(pen)
            self.scene.addItem(line)
            
            # 添加角度标签
            label_dist = self.max_range + 500
            x_label = label_dist * np.cos(rad)
            y_label = label_dist * np.sin(rad)
            
            label = QGraphicsTextItem(f'{angle}°')
            label.setDefaultTextColor(QColor(150, 255, 150))
            label.setFont(QFont("Arial", 150))
            
            # 居中对齐
            bounds = label.boundingRect()
            label.setPos(x_label - bounds.width()/2, y_label - bounds.height()/2)
            self.scene.addItem(label)
            
        # 添加中心十字线
        pen = QPen(QColor(0, 200, 0), 0)
        pen.setStyle(Qt.SolidLine)
        
        h_line = QGraphicsLineItem(-1000, 0, 1000, 0)
        h_line.setPen(pen)
        self.scene.addItem(h_line)
        
        v_line = QGraphicsLineItem(0, -1000, 0, 1000)
        v_line.setPen(pen)
        self.scene.addItem(v_line)
        
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1/factor, 1/factor)
        event.accept()
        
    def showEvent(self, event):
        """首次显示时适应视图"""
        if not hasattr(self, '_initial_fit'):
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._initial_fit = True
        super().showEvent(event)
        
    def update_display(self, data, show_tracks=False, track_data=None, initiated_tracks=None):
        """更新显示"""
        # 清除所有显示内容
        self.clear_all_display()
        
        if data is None or data.empty:
            return
            
        # 显示数据点
        for _, row in data.iterrows():
            # 坐标转换
            r = row['range_out']
            azim = np.radians(row['azim_out'])
            x = r * np.sin(azim)
            y = -r * np.cos(azim)
            
            # 确定点大小
            if 'xgb_probability' in row:
                size = 250 if row['xgb_probability'] > 0.8 else 200
            else:
                size = 200 if row.get('SNR/10', 10) > 15 else 150
                
            # 创建点
            point = PointItem(x, y, size, row.to_dict())
            point.point_clicked.connect(self._on_point_clicked)
            self.scene.addItem(point)
            self.point_items.append(point)
            
        # 显示航迹
        if show_tracks and track_data:
            self._draw_tracks(track_data, QColor(0, 255, 255), "原始")
            
        if initiated_tracks:
            self._draw_initiated_tracks(initiated_tracks)
            
    def _draw_tracks(self, tracks, color, label_prefix):
        """绘制航迹"""
        pen = QPen(color, 50)
        
        for track_id, points in tracks.items():
            if len(points) < 2:
                continue
                
            path = QPainterPath()
            first = True
            
            for point in points:
                if 'x' in point and 'y' in point:
                    x, y = point['x'], point['y']
                else:
                    r = point['range_out']
                    azim = np.radians(point['azim_out'])
                    x = r * np.sin(azim)
                    y = -r * np.cos(azim)
                    
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
                    
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            self.scene.addItem(item)
            self.track_items.append(item)
            
    def _draw_initiated_tracks(self, tracks):
        """绘制起批航迹"""
        for track_id, info in tracks.items():
            points = info['points']
            if len(points) < 2:
                continue
                
            # 根据航迹状态选择颜色
            mode = info.get('established_mode', 1)
            if mode == 2:  # 确认
                pen = QPen(QColor(0, 255, 0), 80)  # 绿色实线
                label_color = QColor(0, 255, 0)
                status_text = "确认"
            else:  # 待定
                pen = QPen(QColor(255, 255, 0), 60)  # 黄色
                pen.setStyle(Qt.DashLine)  # 虚线
                label_color = QColor(255, 255, 0)
                status_text = "待定"
                
            path = QPainterPath()
            path.moveTo(points[0]['x'], points[0]['y'])
            
            for pt in points[1:]:
                path.lineTo(pt['x'], pt['y'])
                
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            self.scene.addItem(item)
            self.track_items.append(item)
            
            # 添加标签
            last = points[-1]
            label = QGraphicsTextItem(f"T{track_id}({status_text})")
            label.setDefaultTextColor(label_color)
            label.setFont(QFont("Arial", 100, QFont.Bold))
            label.setPos(last['x'] + 100, last['y'])
            self.scene.addItem(label)
            self.track_items.append(label)
            
    def clear_points(self):
        """清除点"""
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        
        if self.highlight_item:
            self.scene.removeItem(self.highlight_item)
            self.highlight_item = None
            
    def clear_all_display(self):
        """清除所有显示"""
        self.clear_points()
        
        for item in self.track_items:
            self.scene.removeItem(item)
        self.track_items.clear()
        
    def highlight_point(self, point_item):
        """高亮点"""
        if self.highlight_item:
            self.scene.removeItem(self.highlight_item)
            
        if point_item:
            size = point_item.boundingRect().width() * 1.5
            pen = QPen(QColor(255, 255, 0), 80)
            
            highlight = QGraphicsEllipseItem(-size/2, -size/2, size, size)
            highlight.setPos(point_item.pos())
            highlight.setPen(pen)
            highlight.setBrush(QBrush(Qt.NoBrush))
            
            self.scene.addItem(highlight)
            self.highlight_item = highlight
            
    def _on_point_clicked(self, point_item):
        """处理点击"""
        self.highlight_point(point_item)
        self.point_clicked.emit(point_item.point_data)
        
    def display_original_points(self, data):
        """显示原始数据点"""
        self.clear_points()
        
        if data is None or data.empty:
            return
            
        for _, row in data.iterrows():
            # 坐标转换
            r = row['range_out']
            azim = np.radians(row['azim_out'])
            x = r * np.sin(azim)
            y = -r * np.cos(azim)
            
            # 确定点大小
            if 'xgb_probability' in row:
                size = 250 if row['xgb_probability'] > 0.8 else 200
            else:
                size = 200 if row.get('SNR/10', 10) > 15 else 150
                
            # 创建点
            point = PointItem(x, y, size, row.to_dict())
            point.point_clicked.connect(self._on_point_clicked)
            self.scene.addItem(point)
            self.point_items.append(point)
            
    def display_confirmed_track(self, points, track_id):
        """显示确认航迹"""
        if len(points) < 2:
            return
            
        pen = QPen(QColor(0, 255, 0), 80)  # 绿色实线
        path = QPainterPath()
        path.moveTo(points[0]['x'], points[0]['y'])
        
        for pt in points[1:]:
            path.lineTo(pt['x'], pt['y'])
            
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        self.scene.addItem(item)
        self.track_items.append(item)
        
        # 添加标签
        last = points[-1]
        label = QGraphicsTextItem(f"T{track_id}(确认)")
        label.setDefaultTextColor(QColor(0, 255, 0))
        label.setFont(QFont("Arial", 100, QFont.Bold))
        label.setPos(last['x'] + 100, last['y'])
        self.scene.addItem(label)
        self.track_items.append(label)
        
    def display_prepare_track(self, points, track_id):
        """显示预备航迹"""
        if len(points) < 2:
            return
            
        pen = QPen(QColor(255, 255, 0), 60)  # 黄色虚线
        pen.setStyle(Qt.DashLine)
        path = QPainterPath()
        path.moveTo(points[0]['x'], points[0]['y'])
        
        for pt in points[1:]:
            path.lineTo(pt['x'], pt['y'])
            
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        self.scene.addItem(item)
        self.track_items.append(item)
        
        # 添加标签
        last = points[-1]
        label = QGraphicsTextItem(f"T{track_id}(预备)")
        label.setDefaultTextColor(QColor(255, 255, 0))
        label.setFont(QFont("Arial", 100, QFont.Bold))
        label.setPos(last['x'] + 100, last['y'])
        self.scene.addItem(label)
        self.track_items.append(label)
        
    def display_tentative_track(self, points, track_id):
        """显示待定航迹"""
        if len(points) < 2:
            return
            
        pen = QPen(QColor(255, 100, 100), 40)  # 红色细线
        pen.setStyle(Qt.DotLine)
        path = QPainterPath()
        path.moveTo(points[0]['x'], points[0]['y'])
        
        for pt in points[1:]:
            path.lineTo(pt['x'], pt['y'])
            
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        self.scene.addItem(item)
        self.track_items.append(item)
        
        # 添加标签
        last = points[-1]
        label = QGraphicsTextItem(f"T{track_id}(待定)")
        label.setDefaultTextColor(QColor(255, 100, 100))
        label.setFont(QFont("Arial", 80, QFont.Normal))
        label.setPos(last['x'] + 100, last['y'])
        self.scene.addItem(label)
        self.track_items.append(label)
        
    def display_original_track(self, points, track_id):
        """显示原始航迹（从track文件）"""
        if len(points) < 2:
            return
            
        # 使用青色显示原始航迹
        pen = QPen(QColor(0, 255, 255), 60)  # 青色
        pen.setStyle(Qt.DashDotLine)  # 点划线
        path = QPainterPath()
        path.moveTo(points[0]['x'], points[0]['y'])
        
        for pt in points[1:]:
            path.lineTo(pt['x'], pt['y'])
            
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        self.scene.addItem(item)
        self.track_items.append(item)
        
        # 添加标签
        last = points[-1]
        label = QGraphicsTextItem(f"原始T{track_id}")
        label.setDefaultTextColor(QColor(0, 255, 255))
        label.setFont(QFont("Arial", 90, QFont.Bold))
        label.setPos(last['x'] + 100, last['y'])
        self.scene.addItem(label)
        self.track_items.append(label)