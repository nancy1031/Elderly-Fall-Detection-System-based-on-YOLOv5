import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QSlider, QStatusBar, QGroupBox, QFormLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO

# 配置你的模型路径
MODEL_PATH = "runs/detect/fall_det_train/weights/best.pt"

class FallDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能跌倒识别系统 (YOLOv8 + PyQt5)")
        self.setGeometry(100, 100, 1200, 700)

        # 初始化模型
        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 初始化参数
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.is_camera = False

        # 设置主界面布局
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：视频/图像显示区 ---
        self.original_label = QLabel("原始画面")
        self.result_label = QLabel("检测结果")
        for label in [self.original_label, self.result_label]:
            label.setFixedSize(500, 500)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 2px solid #3498db; background-color: #2c3e50; color: white; font-size: 18px;")
        main_layout.addLayout(self._create_display_layout())

        # --- 右侧：控制面板 ---
        control_panel = QGroupBox("系统控制台")
        control_layout = QFormLayout()
        control_layout.setSpacing(15)

        # 按钮区
        self.btn_img = QPushButton("📷 读取图片")
        self.btn_vid = QPushButton("🎬 读取视频")
        self.btn_cam = QPushButton("📹 摄像头监控")
        self.btn_stop = QPushButton("⏹ 停止运行")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white;")

        for btn in [self.btn_img, self.btn_vid, self.btn_cam, self.btn_stop]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("font-size: 14px; border-radius: 5px;")

        # 滑动条区
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_iou)

        control_layout.addRow(self.btn_img)
        control_layout.addRow(self.btn_vid)
        control_layout.addRow(self.btn_cam)
        control_layout.addRow(self.btn_stop)
        control_layout.addRow("置信度阈值 (Conf):", self.conf_slider)
        control_layout.addRow("IOU 阈值:", self.iou_slider)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # 绑定按钮事件
        self.btn_img.clicked.connect(self.load_image)
        self.btn_vid.clicked.connect(self.load_video)
        self.btn_cam.clicked.connect(self.load_camera)
        self.btn_stop.clicked.connect(self.stop_stream)

        # 状态栏
        self.statusBar().showMessage("系统就绪，请选择输入源...")

    def _create_display_layout(self):
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.original_label)
        top_layout.addWidget(self.result_label)
        layout.addLayout(top_layout)
        return layout

    # --- 核心功能：更新帧 ---
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.stop_stream()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            return

        # 模型推理
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        annotated_frame = results[0].plot()

        # 统计跌倒人数
        fall_count = len(results[0].boxes)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.statusBar().showMessage(f"FPS: {fps:.1f} | 检测到跌倒人数: {fall_count}")

        # 更新界面图像
        self.update_label(self.original_label, frame)
        self.update_label(self.result_label, annotated_frame)

    def update_label(self, label, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(500, 500, Qt.KeepAspectRatio))

    # --- 输入源控制 ---
    def load_image(self):
        self.stop_stream()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            frame = cv2.imread(file_path)
            results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)
            annotated_frame = results[0].plot()
            self.update_label(self.original_label, frame)
            self.update_label(self.result_label, annotated_frame)
            self.statusBar().showMessage(f"图片检测完成 | 跌倒人数: {len(results[0].boxes)}")

    def load_video(self):
        self.stop_stream()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.is_camera = False
            self.timer.start(30)  # 约33 FPS

    def load_camera(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_camera = True
            self.timer.start(30)
            self.statusBar().showMessage("摄像头已开启，实时监控中...")
        else:
            self.statusBar().showMessage("错误：无法打开摄像头！")

    def stop_stream(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.original_label.clear()
        self.original_label.setText("原始画面")
        self.result_label.clear()
        self.result_label.setText("检测结果")
        self.statusBar().showMessage("已停止运行")

    # --- 参数调节 ---
    def update_conf(self, value):
        self.conf_thres = value / 100.0

    def update_iou(self, value):
        self.iou_thres = value / 100.0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = FallDetectionGUI()
    window.show()
    sys.exit(app.exec_())