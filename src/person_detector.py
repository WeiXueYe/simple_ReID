"""
人物检测模块
使用YOLOv8进行人物检测
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
from pathlib import Path

# 尝试导入YOLOv8，如果失败则提供友好的错误信息
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics未安装，人物检测功能将不可用")

from .config import Config
from .utils import validate_bbox, logger


class PersonDetector:
    """
    人物检测器类
    使用YOLOv8模型进行人物检测
    """

    def __init__(self, config: Config = None):
        """
        初始化人物检测器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.model = None
        self.device = None
        self.is_initialized = False

        # 检查ultralytics是否可用
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics未安装，请运行: pip install ultralytics"
            )

    def initialize(self, model_path: str = None) -> bool:
        """
        初始化YOLOv8模型

        Args:
            model_path: 模型路径，None则使用默认模型

        Returns:
            是否成功初始化
        """
        try:
            # 确定设备
            self._setup_device()

            # 获取模型路径
            if model_path is None:
                model_path = self._get_default_model_path()

            # 检查模型文件是否存在，不存在则下载
            if not os.path.exists(model_path):
                logger.info(f"模型文件不存在，将自动下载: {model_path}")
                model_path = self._download_model()

            # 加载模型
            logger.info(f"正在加载YOLOv8模型: {model_path}")
            self.model = YOLO(model_path)

            # 移动到指定设备
            if self.device:
                self.model.to(self.device)

            self.is_initialized = True
            logger.info(f"模型初始化成功，设备: {self.device}")

            return True

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.is_initialized = False
            return False

    def _setup_device(self) -> None:
        """
        设置计算设备
        """
        try:
            import torch

            if self.config.USE_GPU and torch.cuda.is_available():
                self.device = f'cuda:{self.config.GPU_DEVICE}'
                logger.info(f"使用GPU: {self.device}")
            else:
                self.device = 'cpu'
                logger.info("使用CPU进行推理")

        except ImportError:
            self.device = 'cpu'
            logger.warning("PyTorch未安装，使用CPU模式")

    def _get_default_model_path(self) -> str:
        """
        获取默认模型路径

        Returns:
            模型文件路径
        """
        model_name = f'yolov8{self.config.YOLO_MODEL_SIZE}.pt'
        return os.path.join(self.config.MODEL_DIR, model_name)

    def _download_model(self) -> str:
        """
        下载YOLOv8预训练模型

        Returns:
            下载后的模型路径
        """
        try:
            # 确保模型目录存在
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)

            model_name = f'yolov8{self.config.YOLO_MODEL_SIZE}'
            logger.info(f"下载模型: {model_name}")

            # 使用ultralytics自动下载
            model = YOLO(f'{model_name}.pt')

            # 保存到指定路径
            model_path = self._get_default_model_path()
            # YOLO模型会自动缓存，我们只需要返回标准路径

            logger.info(f"模型下载完成: {model_path}")
            return f'{model_name}.pt'  # ultralytics使用缓存路径

        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            # 尝试使用在线模型
            model_name = f'yolov8{self.config.YOLO_MODEL_SIZE}'
            logger.info(f"尝试使用在线模型: {model_name}")
            return f'{model_name}.pt'

    def detect_persons(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """
        检测图像中的人物

        Args:
            image: 输入图像 (H, W, C) RGB格式
            confidence_threshold: 置信度阈值，None则使用配置中的默认值

        Returns:
            检测到的人物列表，每个元素包含bbox、confidence等信息
        """
        if not self.is_initialized:
            raise RuntimeError("检测器未初始化，请先调用initialize()")

        if image is None or image.size == 0:
            raise ValueError("输入图像无效")

        confidence_threshold = confidence_threshold or self.config.DETECTION_CONFIDENCE_THRESHOLD

        try:
            # 执行推理
            results = self.model(
                image,
                conf=confidence_threshold,
                classes=self.config.DETECT_CLASSES,
                imgsz=self.config.YOLO_IMG_SIZE,
                device=self.device,
                verbose=False
            )

            # 解析结果
            detections = []
            image_shape = image.shape[:2]  # (height, width)

            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # 获取边界框坐标
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox)

                    # 验证边界框
                    if not validate_bbox((x1, y1, x2, y2), image_shape):
                        continue

                    # 获取置信度
                    confidence = float(boxes.conf[i].cpu().numpy())

                    # 获取类别
                    class_id = int(boxes.cls[i].cpu().numpy())

                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person',
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    }

                    detections.append(detection)

            logger.debug(f"检测到 {len(detections)} 个人物")
            return detections

        except Exception as e:
            logger.error(f"人物检测失败: {e}")
            return []

    def detect_persons_batch(self, images: List[np.ndarray],
                           confidence_threshold: float = None) -> List[List[Dict[str, Any]]]:
        """
        批量检测图像中的人物

        Args:
            images: 图像列表
            confidence_threshold: 置信度阈值

        Returns:
            每张图像的检测结果列表
        """
        if not self.is_initialized:
            raise RuntimeError("检测器未初始化，请先调用initialize()")

        if not images:
            return []

        confidence_threshold = confidence_threshold or self.config.DETECTION_CONFIDENCE_THRESHOLD

        try:
            # 批量推理
            results = self.model(
                images,
                conf=confidence_threshold,
                classes=self.config.DETECT_CLASSES,
                imgsz=self.config.YOLO_IMG_SIZE,
                device=self.device,
                verbose=False,
                batch_size=self.config.BATCH_SIZE
            )

            # 解析批量结果
            all_detections = []

            for result in results:
                detections = []
                boxes = result.boxes
                image_shape = (result.orig_shape[0], result.orig_shape[1])

                for i in range(len(boxes)):
                    # 获取边界框坐标
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)

                    # 验证边界框
                    if not validate_bbox((x1, y1, x2, y2), image_shape):
                        continue

                    # 获取置信度和类别
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())

                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person',
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    }

                    detections.append(detection)

                all_detections.append(detections)

            logger.debug(f"批量检测到 {sum(len(d) for d in all_detections)} 个人物")
            return all_detections

        except Exception as e:
            logger.error(f"批量人物检测失败: {e}")
            return [[] for _ in images]

    def filter_detections(self, detections: List[Dict[str, Any]],
                         min_area: int = None, max_area: int = None,
                         confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """
        过滤检测结果
        增强版本：增加非极大值抑制和面积过滤

        Args:
            detections: 检测结果列表
            min_area: 最小面积
            max_area: 最大面积
            confidence_threshold: 置信度阈值

        Returns:
            过滤后的结果
        """
        if not detections:
            return []

        confidence_threshold = confidence_threshold or self.config.DETECTION_CONFIDENCE_THRESHOLD

        # 1. 置信度预过滤
        confident_detections = [
            d for d in detections if d['confidence'] >= confidence_threshold * 0.8
        ]

        if not confident_detections:
            return []

        # 2. 面积过滤
        min_area = min_area or 100  # 最小面积阈值
        max_area = max_area or 100000  # 最大面积阈值

        area_filtered = []
        for detection in confident_detections:
            area = detection['area']
            if min_area <= area <= max_area:
                area_filtered.append(detection)

        if not area_filtered:
            return []

        # 3. 非极大值抑制（NMS）
        final_detections = self._non_max_suppression(area_filtered, iou_threshold=0.3)

        # 4. 最终置信度过滤
        filtered = [
            d for d in final_detections if d['confidence'] >= confidence_threshold
        ]

        # 按置信度排序
        filtered.sort(key=lambda x: x['confidence'], reverse=True)

        return filtered

    def _non_max_suppression(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        非极大值抑制，去除重叠的检测框

        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值

        Returns:
            去重后的结果
        """
        if len(detections) <= 1:
            return detections

        # 按置信度排序
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while sorted_detections:
            # 选择置信度最高的
            current = sorted_detections.pop(0)
            keep.append(current)

            # 移除与当前检测框重叠度过高的其他检测框
            remaining = []
            for detection in sorted_detections:
                iou = self._calculate_iou(current['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining.append(detection)

            sorted_detections = remaining

        return keep

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的IoU
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集区域
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # 检查是否有交集
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # 计算IoU
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def get_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取检测结果统计信息

        Args:
            detections: 检测结果列表

        Returns:
            统计信息
        """
        if not detections:
            return {
                'count': 0,
                'avg_confidence': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0
            }

        confidences = [d['confidence'] for d in detections]
        areas = [d['area'] for d in detections]

        return {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_area': np.mean(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas)
        }

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]],
                           show_confidence: bool = True) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            detections: 检测结果
            show_confidence: 是否显示置信度

        Returns:
            可视化后的图像
        """
        # 创建图像副本
        vis_image = image.copy()

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            if show_confidence:
                label = f"Person {i+1}: {confidence:.2f}"
            else:
                label = f"Person {i+1}"

            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)

            # 绘制标签文本
            cv2.putText(vis_image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis_image

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        if not self.is_initialized or self.model is None:
            return {}

        try:
            return {
                'model_type': 'YOLOv8',
                'model_size': self.config.YOLO_MODEL_SIZE,
                'device': self.device,
                'img_size': self.config.YOLO_IMG_SIZE,
                'confidence_threshold': self.config.DETECTION_CONFIDENCE_THRESHOLD,
                'detect_classes': self.config.DETECT_CLASSES
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}

    def __del__(self):
        """析构函数"""
        if self.model is not None:
            try:
                # 清理模型
                del self.model
                self.model = None
            except:
                pass


def main():
    """
    测试函数
    """
    # 创建检测器
    detector = PersonDetector()

    try:
        # 初始化模型
        if detector.initialize():
            print("模型初始化成功")

            # 获取模型信息
            model_info = detector.get_model_info()
            print("模型信息:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            # 测试图像（这里需要替换为实际的图像文件）
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色图像用于测试

            # 执行检测
            detections = detector.detect_persons(test_image)
            print(f"\n检测到 {len(detections)} 个人物")

            for i, det in enumerate(detections):
                print(f"  人物 {i+1}: {det}")

        else:
            print("模型初始化失败")

    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()