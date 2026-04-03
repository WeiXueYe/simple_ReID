"""
通用工具函数模块
包含图像处理、相似度计算、文件IO、时间转换等功能
"""

import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 图像处理工具函数 ====================

def resize_and_pad(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    将图像resize到目标尺寸，保持宽高比并使用padding填充

    Args:
        image: 输入图像 (H, W, C)
        target_size: 目标尺寸 (width, height)

    Returns:
        处理后的图像
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建目标图像并填充
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
    else:
        padded = np.zeros((target_h, target_w), dtype=np.uint8)

    # 计算padding位置
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # 放置缩放后的图像
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


def normalize_image(image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """
    图像归一化处理

    Args:
        image: 输入图像
        mean: 均值，默认ImageNet均值
        std: 标准差，默认ImageNet标准差

    Returns:
        归一化后的图像
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # 转换为float32并归一化到[0,1]
    image = image.astype(np.float32) / 255.0

    # 减去均值，除以标准差
    image[..., 0] = (image[..., 0] - mean[0]) / std[0]
    image[..., 1] = (image[..., 1] - mean[1]) / std[1]
    image[..., 2] = (image[..., 2] - mean[2]) / std[2]

    return image


def crop_person(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    根据边界框裁剪人物区域

    Args:
        image: 输入图像
        bbox: 边界框 (x1, y1, x2, y2)

    Returns:
        裁剪后的人物图像
    """
    x1, y1, x2, y2 = bbox

    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效的边界框: {bbox}")

    return image[y1:y2, x1:x2]


def preprocess_for_reid(image: np.ndarray, target_size: Tuple[int, int] = (128, 256)) -> np.ndarray:
    """
    为ReID模型预处理图像

    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)

    Returns:
        预处理后的图像
    """
    # 调整大小并保持宽高比
    resized = resize_and_pad(image, target_size)

    # 归一化
    normalized = normalize_image(resized)

    return normalized


# ==================== 相似度计算函数 ====================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        余弦相似度值（0-1之间）
    """
    # 确保向量是1维的
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算模长
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)

    # 确保结果在[-1, 1]范围内（处理浮点误差）
    similarity = np.clip(similarity, -1.0, 1.0)

    return float(similarity)


def batch_cosine_similarity(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """
    批量计算余弦相似度矩阵

    Args:
        query_features: 查询特征矩阵 (N, D)
        gallery_features: 库特征矩阵 (M, D)

    Returns:
        相似度矩阵 (N, M)
    """
    # 归一化特征向量
    query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_norm = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    # 计算相似度矩阵
    similarity_matrix = np.dot(query_norm, gallery_norm.T)

    return similarity_matrix


def find_best_match(query_feature: np.ndarray, gallery_features: Dict[str, np.ndarray],
                   threshold: float = 0.7) -> Tuple[Optional[str], float]:
    """
    在特征库中找到最佳匹配

    Args:
        query_feature: 查询特征
        gallery_features: 特征库 {id: feature}
        threshold: 相似度阈值

    Returns:
        (最佳匹配ID, 相似度分数)，如果没有匹配则返回(None, 0.0)
    """
    if not gallery_features:
        return None, 0.0

    best_id = None
    best_score = 0.0

    for person_id, gallery_feature in gallery_features.items():
        score = cosine_similarity(query_feature, gallery_feature)
        if score > best_score and score >= threshold:
            best_score = score
            best_id = person_id

    return best_id, best_score


# ==================== 文件IO操作函数 ====================

def save_json(data: Dict[str, Any], filepath: str, ensure_ascii: bool = False) -> None:
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
        ensure_ascii: 是否确保ASCII编码
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)

        logger.info(f"成功保存JSON文件: {filepath}")
    except Exception as e:
        logger.error(f"保存JSON文件失败 {filepath}: {e}")
        raise


def load_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"成功加载JSON文件: {filepath}")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件失败 {filepath}: {e}")
        raise


def get_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    获取指定目录下的视频文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，默认['.mp4', '.avi', '.mov']

    Returns:
        视频文件路径列表
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']

    video_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.warning(f"目录不存在: {directory}")
        return video_files

    for ext in extensions:
        video_files.extend([str(f) for f in directory_path.glob(f"*{ext}")])

    video_files.sort()
    logger.info(f"在 {directory} 中找到 {len(video_files)} 个视频文件")

    return video_files


def create_directory(directory: str) -> None:
    """
    创建目录（如果不存在）

    Args:
        directory: 目录路径
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"确保目录存在: {directory}")
    except Exception as e:
        logger.error(f"创建目录失败 {directory}: {e}")
        raise


# ==================== 时间格式转换函数 ====================

def frame_to_timestamp(frame_number: int, fps: float) -> int:
    """
    将帧号转换为时间戳（毫秒）

    Args:
        frame_number: 帧号
        fps: 帧率

    Returns:
        时间戳（毫秒）
    """
    if fps <= 0:
        raise ValueError("帧率必须大于0")

    timestamp_ms = int((frame_number / fps) * 1000)
    return timestamp_ms


def timestamp_to_frame(timestamp_ms: int, fps: float) -> int:
    """
    将时间戳转换为帧号

    Args:
        timestamp_ms: 时间戳（毫秒）
        fps: 帧率

    Returns:
        帧号
    """
    if fps <= 0:
        raise ValueError("帧率必须大于0")

    frame_number = int((timestamp_ms / 1000) * fps)
    return frame_number


def frame_to_timecode(frame_number: int, fps: float) -> str:
    """
    将帧号转换为时间码格式 (HH:MM:SS.mmm)

    Args:
        frame_number: 帧号
        fps: 帧率

    Returns:
        时间码字符串
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def timecode_to_frame(timecode: str, fps: float) -> int:
    """
    将时间码转换为帧号

    Args:
        timecode: 时间码字符串 (HH:MM:SS.mmm)
        fps: 帧率

    Returns:
        帧号
    """
    try:
        # 解析时间码
        parts = timecode.split(':')
        if len(parts) != 3:
            raise ValueError("时间码格式错误")

        hours = int(parts[0])
        minutes = int(parts[1])

        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        frame_number = int(total_seconds * fps)

        return frame_number
    except Exception as e:
        logger.error(f"时间码解析失败: {timecode}, 错误: {e}")
        raise


def format_duration(milliseconds: int) -> str:
    """
    格式化时长显示

    Args:
        milliseconds: 毫秒数

    Returns:
        格式化的时长字符串
    """
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60

    if hours > 0:
        return f"{hours}小时{minutes%60}分{seconds%60}秒"
    elif minutes > 0:
        return f"{minutes}分{seconds%60}秒"
    else:
        return f"{seconds}秒"


# ==================== 其他工具函数 ====================

def generate_person_id() -> str:
    """
    生成新的人物ID

    Returns:
        人物ID字符串
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"person_{timestamp}"


def validate_bbox(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> bool:
    """
    验证边界框的有效性

    Args:
        bbox: 边界框 (x1, y1, x2, y2)
        image_shape: 图像形状 (height, width)

    Returns:
        是否有效
    """
    x1, y1, x2, y2 = bbox
    height, width = image_shape

    # 检查坐标有效性
    if x1 >= x2 or y1 >= y2:
        return False

    # 检查是否在图像范围内
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False

    # 检查面积是否过小
    area = (x2 - x1) * (y2 - y1)
    if area < 100:  # 最小面积阈值
        return False

    return True


def merge_time_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    合并重叠或相邻的时间段

    Args:
        segments: 时间段列表 [(start, end), ...]

    Returns:
        合并后的时间段列表
    """
    if not segments:
        return []

    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda x: x[0])

    merged = [sorted_segments[0]]

    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]

        # 如果当前段与前一段重叠或相邻，则合并
        if current_start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    计算两个边界框的IoU（交并比）

    Args:
        bbox1: 边界框1 (x1, y1, x2, y2)
        bbox2: 边界框2 (x1, y1, x2, y2)

    Returns:
        IoU值
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

    # 计算面积
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou


def detect_shot_transition(prev_frame: np.ndarray, curr_frame: np.ndarray,
                          threshold: float = 0.3) -> bool:
    """
    检测镜头切换

    Args:
        prev_frame: 前一帧图像
        curr_frame: 当前帧图像
        threshold: 差异阈值

    Returns:
        是否检测到镜头切换
    """
    if prev_frame is None or curr_frame is None:
        return False

    try:
        # 转换为灰度图
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame

        # 计算直方图差异
        prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])

        # 归一化
        prev_hist = prev_hist / prev_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        # 计算直方图差异
        hist_diff = np.sum(np.abs(prev_hist - curr_hist)) / 2

        # 计算像素级差异
        pixel_diff = np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))) / 255

        # 综合差异度
        combined_diff = 0.7 * hist_diff + 0.3 * pixel_diff

        return combined_diff > threshold

    except Exception as e:
        logger.warning(f"镜头切换检测失败: {e}")
        return False


def predict_person_position(bbox_history: List[Tuple[int, int, int, int]],
                          num_predictions: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    基于历史轨迹预测人物位置

    Args:
        bbox_history: 边界框历史 [(x1, y1, x2, y2), ...]
        num_predictions: 预测帧数

    Returns:
        预测的位置列表
    """
    if len(bbox_history) < 2:
        return []

    try:
        # 提取中心点轨迹
        centers = []
        for bbox in bbox_history:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))

        # 计算运动趋势
        if len(centers) >= 2:
            dx = centers[-1][0] - centers[-2][0]
            dy = centers[-1][1] - centers[-2][1]
        else:
            dx, dy = 0, 0

        # 预测未来位置
        predictions = []
        last_bbox = bbox_history[-1]
        width = last_bbox[2] - last_bbox[0]
        height = last_bbox[3] - last_bbox[1]

        for i in range(1, num_predictions + 1):
            pred_x = last_bbox[0] + dx * i
            pred_y = last_bbox[1] + dy * i
            pred_bbox = (
                int(pred_x),
                int(pred_y),
                int(pred_x + width),
                int(pred_y + height)
            )
            predictions.append(pred_bbox)

        return predictions

    except Exception as e:
        logger.warning(f"位置预测失败: {e}")
        return []