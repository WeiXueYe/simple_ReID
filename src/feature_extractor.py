"""
特征提取模块
使用预训练的人体ReID模型提取人物特征
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
from pathlib import Path
from collections import OrderedDict
import hashlib

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 尝试导入torchreid，如果失败则提供友好的错误信息
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logging.warning("TorchReID未安装，特征提取功能将不可用")

from .config import Config
from .utils import preprocess_for_reid, logger


class FeatureExtractor:
    """
    特征提取器类
    使用预训练的ReID模型提取人物特征
    """

    def __init__(self, config: Config = None):
        """
        初始化特征提取器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.model = None
        self.device = None
        self.is_initialized = False
        self.feature_cache = OrderedDict()
        self.cache_size = self.config.FEATURE_CACHE_SIZE
        self._setup_device()  # 提前设置设备

        # 检查torchreid是否可用
        if not TORCHREID_AVAILABLE:
            raise ImportError(
                "TorchReID未安装，请运行: pip install torchreid"
            )

    def initialize(self, model_name: str = None) -> bool:
        """
        初始化ReID模型

        Args:
            model_name: 模型名称，None则使用配置中的默认值

        Returns:
            是否成功初始化
        """
        try:
            # 设置设备
            self._setup_device()

            # 获取模型名称
            model_name = model_name or self.config.REID_MODEL_NAME

            logger.info(f"正在初始化ReID模型: {model_name}")

            # 创建模型
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,  # 不需要分类，只提取特征
                loss='softmax',
                pretrained=True,
                use_gpu=self.config.USE_GPU
            )

            # 设置为评估模式
            self.model.eval()

            # 移动到指定设备
            self.model.to(self.device)

            # 启用cudnn benchmark以提高GPU性能
            if torch.cuda.is_available() and self.config.USE_GPU:
                torch.backends.cudnn.benchmark = True
                logger.info("启用cudnn benchmark优化")



            self.is_initialized = True
            logger.info(f"ReID模型初始化成功，设备: {self.device}")

            return True

        except Exception as e:
            logger.error(f"ReID模型初始化失败: {e}")
            self.is_initialized = False
            return False

    def _setup_device(self) -> None:
        """
        设置计算设备
        """
        try:
            if self.config.USE_GPU and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.config.GPU_DEVICE}')
                logger.info(f"使用GPU: {self.device}")
            else:
                self.device = torch.device('cpu')
                logger.info("使用CPU进行推理")

        except Exception as e:
            self.device = torch.device('cpu')
            logger.warning(f"设备设置失败，使用CPU: {e}")

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理单张图像

        Args:
            image: 输入图像 (H, W, C) RGB格式

        Returns:
            预处理后的张量
        """
        # 使用通用预处理
        processed = preprocess_for_reid(
            image,
            target_size=(self.config.REID_IMG_WIDTH, self.config.REID_IMG_HEIGHT)
        )

        # 转换为张量
        tensor = torch.from_numpy(processed).permute(2, 0, 1).float()

        # 添加批次维度
        tensor = tensor.unsqueeze(0)

        return tensor

    def _preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        预处理批量图像

        Args:
            images: 图像列表

        Returns:
            预处理后的张量批次
        """
        processed_images = []

        for image in images:
            processed = preprocess_for_reid(
                image,
                target_size=(self.config.REID_IMG_WIDTH, self.config.REID_IMG_HEIGHT)
            )
            tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
            processed_images.append(tensor)

        # 堆叠成批次
        batch = torch.stack(processed_images, dim=0)

        return batch

    def extract_features(self, images: List[np.ndarray], use_cache: bool = True) -> List[np.ndarray]:
        """
        提取人物特征

        Args:
            images: 人物图像列表
            use_cache: 是否使用缓存

        Returns:
            特征向量列表
        """
        if not self.is_initialized:
            raise RuntimeError("特征提取器未初始化，请先调用initialize()")

        if not images:
            return []

        # 检查缓存
        if use_cache:
            cached_features = []
            uncached_images = []
            uncached_indices = []

            for i, image in enumerate(images):
                cache_key = self._get_cache_key(image)
                if cache_key in self.feature_cache:
                    cached_features.append((i, self.feature_cache[cache_key]))
                else:
                    uncached_images.append(image)
                    uncached_indices.append(i)

            # 如果有缓存的特征，直接使用
            if cached_features:
                all_features = [None] * len(images)
                for idx, feature in cached_features:
                    all_features[idx] = feature

                # 只处理未缓存的图像
                if uncached_images:
                    new_features = self._extract_features_batch(uncached_images)
                    for i, feature in enumerate(new_features):
                        orig_idx = uncached_indices[i]
                        all_features[orig_idx] = feature

                        # 添加到缓存
                        cache_key = self._get_cache_key(uncached_images[i])
                        self._add_to_cache(cache_key, feature)

                return all_features

        # 没有缓存或不需要缓存，直接处理所有图像
        features = self._extract_features_batch(images)

        # 添加到缓存
        if use_cache:
            for image, feature in zip(images, features):
                cache_key = self._get_cache_key(image)
                self._add_to_cache(cache_key, feature)

        return features

    def _extract_features_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量提取特征

        Args:
            images: 图像列表

        Returns:
            特征向量列表
        """
        if not images:
            return []

        try:
            # 确保使用最大批处理大小
            batch_size = min(len(images), self.config.BATCH_SIZE)

            all_features = []

            # 分批处理以提高GPU利用率
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]

                # 预处理
                batch = self._preprocess_batch(batch_images)

                # 移动到设备
                batch = batch.to(self.device)

                # 提取特征
                with torch.no_grad():
                    batch_features = self.model(batch)

                # 立即移回CPU并转换为numpy，释放GPU内存
                batch_features = batch_features.cpu().numpy()

                # 归一化特征向量
                batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)

                all_features.extend([batch_features[j] for j in range(len(batch_images))])

            logger.debug(f"成功提取 {len(all_features)} 个特征")
            return all_features

        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            return []

    def extract_single_feature(self, image: np.ndarray, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        提取单个人物特征

        Args:
            image: 人物图像
            use_cache: 是否使用缓存

        Returns:
            特征向量或None（如果失败）
        """
        if image is None or image.size == 0:
            return None

        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key(image)
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]

        try:
            # 预处理
            tensor = self._preprocess_image(image)

            # 移动到设备
            tensor = tensor.to(self.device)

            # 提取特征
            with torch.no_grad():
                feature = self.model(tensor)

            # 转换为numpy并归一化
            feature = feature.cpu().numpy().flatten()
            feature = feature / np.linalg.norm(feature)

            # 添加到缓存
            if use_cache:
                self._add_to_cache(cache_key, feature)

            return feature

        except Exception as e:
            logger.error(f"单特征提取失败: {e}")
            return None

    def _get_cache_key(self, image: np.ndarray) -> str:
        """
        生成图像缓存键

        Args:
            image: 图像

        Returns:
            缓存键
        """
        # 使用图像数据的哈希作为缓存键
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def _add_to_cache(self, key: str, feature: np.ndarray) -> None:
        """
        添加特征到缓存

        Args:
            key: 缓存键
            feature: 特征向量
        """
        # 如果缓存已满，删除最旧的项
        if len(self.feature_cache) >= self.cache_size:
            self.feature_cache.popitem(last=False)

        # 添加新项
        self.feature_cache[key] = feature.copy()

    def clear_cache(self) -> None:
        """
        清空特征缓存
        """
        self.feature_cache.clear()
        logger.info("特征缓存已清空")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息

        Returns:
            缓存信息字典
        """
        return {
            'cache_size': len(self.feature_cache),
            'max_cache_size': self.cache_size,
            'cache_usage': len(self.feature_cache) / self.cache_size
        }

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
                'model_name': self.config.REID_MODEL_NAME,
                'feature_dim': self.config.FEATURE_DIM,
                'device': str(self.device),
                'img_size': (self.config.REID_IMG_WIDTH, self.config.REID_IMG_HEIGHT),
                'use_gpu': self.config.USE_GPU
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}

    def compute_similarity_matrix(self, features1: List[np.ndarray],
                                features2: List[np.ndarray]) -> np.ndarray:
        """
        计算特征相似度矩阵

        Args:
            features1: 特征列表1
            features2: 特征列表2

        Returns:
            相似度矩阵
        """
        if not features1 or not features2:
            return np.array([])

        # 转换为矩阵
        matrix1 = np.array(features1)
        matrix2 = np.array(features2)

        # 计算余弦相似度
        similarity_matrix = np.dot(matrix1, matrix2.T)

        return similarity_matrix

    def find_similar_features(self, query_feature: np.ndarray,
                            gallery_features: List[np.ndarray],
                            threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        在特征库中查找相似特征

        Args:
            query_feature: 查询特征
            gallery_features: 特征库
            threshold: 相似度阈值

        Returns:
            相似特征索引和相似度列表
        """
        if not gallery_features:
            return []

        # 计算相似度
        similarities = []
        for gallery_feature in gallery_features:
            similarity = np.dot(query_feature, gallery_feature)
            similarities.append(similarity)

        # 过滤并排序
        similar_indices = [
            (i, sim) for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        similar_indices.sort(key=lambda x: x[1], reverse=True)

        return similar_indices

    def __del__(self):
        """析构函数"""
        self.cleanup()

    def cleanup(self) -> None:
        """
        清理资源
        """
        try:
            # 清理模型
            if self.model is not None:
                del self.model
                self.model = None

            # 清理缓存
            self.feature_cache.clear()

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU内存已清理")

        except Exception as e:
            logger.warning(f"清理资源时发生错误: {e}")


def main():
    """
    测试函数
    """
    # 创建特征提取器
    extractor = FeatureExtractor()

    try:
        # 初始化模型
        if extractor.initialize():
            print("ReID模型初始化成功")

            # 获取模型信息
            model_info = extractor.get_model_info()
            print("模型信息:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            # 测试特征提取（使用随机图像）
            test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

            # 提取单个特征
            feature = extractor.extract_single_feature(test_image)
            if feature is not None:
                print(f"\n提取特征成功，维度: {feature.shape}")
                print(f"特征范数: {np.linalg.norm(feature):.6f}")

            # 测试批量特征提取
            test_images = [test_image] * 3
            features = extractor.extract_features(test_images)
            print(f"\n批量提取 {len(features)} 个特征成功")

            # 获取缓存信息
            cache_info = extractor.get_cache_info()
            print(f"\n缓存信息: {cache_info}")

        else:
            print("ReID模型初始化失败")

    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()