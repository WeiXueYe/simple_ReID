#!/usr/bin/env python3
"""
模型升级测试脚本
比较不同ReID模型的性能和精度
"""

import cv2
import numpy as np
import torch
import torchreid
from src.feature_extractor import FeatureExtractor
from src.config import Config
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_performance(model_name, test_images, iterations=10):
    """
    测试模型性能

    Args:
        model_name: 模型名称
        test_images: 测试图像列表
        iterations: 测试迭代次数

    Returns:
        性能统计信息
    """
    print(f"\n=== 测试模型: {model_name} ===")

    # 创建配置
    config = Config()
    config.REID_MODEL_NAME = model_name

    # 初始化特征提取器
    extractor = FeatureExtractor(config)

    try:
        start_time = time.time()
        success = extractor.initialize()
        init_time = time.time() - start_time

        if not success:
            print(f"模型 {model_name} 初始化失败")
            return None

        print(f"初始化时间: {init_time:.3f}秒")

        # 获取模型信息
        model_info = extractor.get_model_info()
        print(f"模型信息: {model_info}")

        # 测试特征提取速度
        times = []
        for i in range(iterations):
            start_time = time.time()
            features = extractor.extract_features(test_images)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = len(test_images) / avg_time if avg_time > 0 else 0

        print(f"平均处理时间: {avg_time:.3f}±{std_time:.3f}秒")
        print(f"处理速度: {fps:.1f} FPS")
        print(f"特征维度: {len(features[0]) if features else 'N/A'}")

        # 测试特征质量（自相似度）
        if features:
            similarities = []
            for i in range(min(3, len(features))):
                sim = np.dot(features[i], features[i])
                similarities.append(sim)
            avg_similarity = np.mean(similarities)
            print(f"自相似度: {avg_similarity:.6f} (理想值: 1.0)")

        # 清理资源
        extractor.cleanup()

        return {
            'model_name': model_name,
            'init_time': init_time,
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': fps,
            'feature_dim': len(features[0]) if features else 0,
            'self_similarity': avg_similarity if features else 0
        }

    except Exception as e:
        print(f"测试模型 {model_name} 时出错: {e}")
        return None

def create_test_images(count=5, size=(256, 128, 3)):
    """创建测试图像"""
    images = []
    for i in range(count):
        # 创建不同颜色的测试图像
        color = (i * 50 % 255, (i * 80) % 255, (i * 120) % 255)
        img = np.full(size, color, dtype=np.uint8)
        images.append(img)
    return images

def compare_models():
    """比较不同模型"""
    # 可用模型列表
    available_models = [
        'se_resnext50_32x4d',  # 当前使用的模型
        'osnet_x1_0',          # OSNet基础版本
        'osnet_ain_x1_0',      # OSNet + Attention
        'osnet_ibn_x1_0',      # OSNet + IBN
        'resnet50',            # 基础ResNet
        'mlfn',                # 多粒度特征学习
    ]

    # 创建测试图像
    test_images = create_test_images(5)

    results = []

    for model_name in available_models:
        try:
            # 检查模型是否可用
            result = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )
            print(f"模型 {model_name} 可用")

            # 测试性能
            perf_result = test_model_performance(model_name, test_images)
            if perf_result:
                results.append(perf_result)

        except Exception as e:
            print(f"模型 {model_name} 不可用: {e}")

    return results

def print_comparison_results(results):
    """打印比较结果"""
    print("\n" + "="*80)
    print("模型性能比较结果")
    print("="*80)

    # 按FPS排序
    sorted_results = sorted(results, key=lambda x: x['fps'], reverse=True)

    print(f"{'模型名称':<20} {'FPS':<8} {'特征维度':<10} {'自相似度':<12} {'初始化时间':<12}")
    print("-" * 80)

    for result in sorted_results:
        print(f"{result['model_name']:<20} {result['fps']:<8.1f} {result['feature_dim']:<10} {result['self_similarity']:<12.6f} {result['init_time']:<12.3f}")

    print("\n推荐选择:")
    if len(sorted_results) >= 2:
        # 性能最好的
        fastest = sorted_results[0]
        print(f"速度最快: {fastest['model_name']} ({fastest['fps']:.1f} FPS)")

        # 特征质量最好的（自相似度最接近1.0）
        best_quality = min(sorted_results, key=lambda x: abs(x['self_similarity'] - 1.0))
        print(f"质量最佳: {best_quality['model_name']} (自相似度: {best_quality['self_similarity']:.6f})")

def main():
    """主函数"""
    print("开始ReID模型性能测试...")

    # 比较模型
    results = compare_models()

    if results:
        print_comparison_results(results)

        # 推荐最佳模型
        best_model = max(results, key=lambda x: x['fps'])
        print(f"\n推荐升级到: {best_model['model_name']}")
        print("原因:")
        print(f"  - 处理速度: {best_model['fps']:.1f} FPS")
        print(f"  - 特征质量: {best_model['self_similarity']:.6f}")

    else:
        print("没有可用的模型进行测试")

if __name__ == "__main__":
    main()