"""
主控制模块
协调所有子模块的工作流程，提供完整的人物ReID处理流程
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import Config
from .video_processor import VideoProcessor
from .person_detector import PersonDetector
from .feature_extractor import FeatureExtractor
from .person_tracker import PersonTracker
from .appearance_analyzer import AppearanceAnalyzer
from .utils import get_video_files, logger


class MainController:
    """
    主控制器类
    协调所有子模块，提供完整的人物ReID处理流程
    """

    def __init__(self, config: Config = None):
        """
        初始化主控制器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.video_processor = None
        self.person_detector = None
        self.feature_extractor = None
        self.person_tracker = None
        self.appearance_analyzer = None
        self.is_initialized = False

        # 处理统计信息
        self.stats = {
            'processed_videos': 0,
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'total_processing_time': 0.0,
            'errors': []
        }

    def initialize(self) -> bool:
        """
        初始化所有子模块

        Returns:
            是否成功初始化
        """
        try:
            logger.info("开始初始化主控制器...")

            # 初始化视频处理器
            logger.info("初始化视频处理器...")
            self.video_processor = VideoProcessor(self.config)

            # 初始化人物检测器
            logger.info("初始化人物检测器...")
            self.person_detector = PersonDetector(self.config)
            if not self.person_detector.initialize():
                raise RuntimeError("人物检测器初始化失败")

            # 初始化特征提取器
            logger.info("初始化特征提取器...")
            self.feature_extractor = FeatureExtractor(self.config)
            if not self.feature_extractor.initialize():
                raise RuntimeError("特征提取器初始化失败")

            # 初始化人物跟踪器
            logger.info("初始化人物跟踪器...")
            self.person_tracker = PersonTracker(self.config)
            if not self.person_tracker.initialize():
                raise RuntimeError("人物跟踪器初始化失败")

            # 初始化出场时间分析器
            logger.info("初始化出场时间分析器...")
            self.appearance_analyzer = AppearanceAnalyzer(self.config)

            self.is_initialized = True
            logger.info("主控制器初始化完成")

            return True

        except Exception as e:
            logger.error(f"主控制器初始化失败: {e}")
            self._cleanup()
            return False

    def process_single_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        处理单个视频文件

        Args:
            video_path: 视频文件路径

        Returns:
            处理结果或None（如果失败）
        """
        if not self.is_initialized:
            raise RuntimeError("主控制器未初始化，请先调用initialize()")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        start_time = time.time()
        logger.info(f"开始处理视频: {video_path}")

        try:
            # 打开视频
            if not self.video_processor.open_video(video_path):
                raise RuntimeError(f"无法打开视频: {video_path}")

            # 获取视频信息
            video_info = self.video_processor.get_video_info()

            # 设置出场分析器的视频信息
            self.appearance_analyzer.set_video_info(video_info)

            # 重置跟踪器状态
            self.person_tracker.reset()

            # 主处理循环
            tracking_results = self._process_video_frames()

            # 分析出场时间
            analysis_result = self.appearance_analyzer.analyze_appearances(tracking_results)

            # 添加视频信息到结果
            analysis_result['video_info'] = video_info

            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['processed_videos'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(f"视频处理完成: {video_path} (耗时: {processing_time:.2f}秒)")

            return analysis_result

        except Exception as e:
            error_msg = f"处理视频失败 {video_path}: {e}"
            logger.error(error_msg)
            self.stats['errors'].append({
                'video': video_path,
                'error': str(e),
                'timestamp': time.time()
            })
            return None

        finally:
            # 清理资源
            if self.video_processor:
                self.video_processor.close()

            # 清理特征缓存
            if self.feature_extractor:
                self.feature_extractor.clear_cache()

    def _process_video_frames(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理视频帧的主循环

        Returns:
            跟踪结果 {person_id: [appearance_records]}
        """
        logger.info("开始处理视频帧...")

        frame_count = 0
        processed_count = 0
        prev_frame = None

        try:
            # 提取视频帧
            for frame, frame_number, timestamp_ms in self.video_processor.extract_frames():
                frame_count = frame_number
                processed_count += 1

                # 检测镜头切换
                if prev_frame is not None and self.config.ENABLE_SHOT_TRANSITION_DETECTION:
                    is_transition = self.person_tracker.detect_and_handle_shot_transition(
                        prev_frame, frame
                    )
                    if is_transition:
                        logger.info(f"帧 {frame_number}: 检测到镜头切换")

                # 处理当前帧
                self._process_single_frame(frame, frame_number, timestamp_ms)

                # 更新前一帧
                prev_frame = frame.copy() if frame is not None else None

                # 每100帧输出一次进度
                if processed_count % 100 == 0:
                    progress = (frame_count / self.video_processor.video_info.get('frame_count', 1)) * 100
                    logger.info(f"处理进度: {progress:.1f}% ({frame_count} 帧)")

            logger.info(f"帧处理完成，共处理 {processed_count} 帧")

        except Exception as e:
            logger.error(f"帧处理过程中发生错误: {e}")
            raise

        # 获取最终的跟踪结果
        tracking_results = self.person_tracker.get_tracking_results()

        # 更新统计信息
        self.stats['total_frames_processed'] += processed_count
        self.stats['total_persons_detected'] += len(tracking_results)

        return tracking_results

    def _process_single_frame(self, frame: Any, frame_number: int, timestamp_ms: int) -> None:
        """
        处理单帧

        Args:
            frame: 帧图像
            frame_number: 帧号
            timestamp_ms: 时间戳（毫秒）
        """
        try:
            # 1. 人物检测
            detections = self.person_detector.detect_persons(frame)

            if not detections:
                # 没有检测到人物，更新跟踪器
                self.person_tracker.update_empty_frame(frame_number)
                return

            # 2. 提取人物图像
            person_images = []
            valid_detections = []

            for detection in detections:
                try:
                    bbox = detection['bbox']
                    person_image = self._crop_person_image(frame, bbox)
                    if person_image is not None:
                        person_images.append(person_image)
                        valid_detections.append(detection)
                except Exception as e:
                    logger.warning(f"裁剪人物图像失败 (帧 {frame_number}): {e}")
                    continue

            if not person_images:
                self.person_tracker.update_empty_frame(frame_number)
                return

            # 3. 批量提取特征（GPU优化）
            features = self.feature_extractor.extract_features(person_images, use_cache=True)

            if not features or len(features) != len(person_images):
                logger.warning(f"特征提取失败或数量不匹配 (帧 {frame_number})")
                self.person_tracker.update_empty_frame(frame_number)
                return

            # 4. 更新跟踪器
            frame_data = []
            for i, (detection, feature) in enumerate(zip(valid_detections, features)):
                person_data = {
                    'bbox': detection['bbox'],
                    'feature': feature,
                    'confidence': detection['confidence'],
                    'frame_number': frame_number,
                    'timestamp_ms': timestamp_ms
                }
                frame_data.append(person_data)

            self.person_tracker.update_frame(frame_data)

        except Exception as e:
            logger.error(f"处理帧 {frame_number} 失败: {e}")
            # 继续处理下一帧，不中断整个流程

    def _crop_person_image(self, frame: Any, bbox: tuple) -> Optional[Any]:
        """
        裁剪人物图像

        Args:
            frame: 原始帧
            bbox: 边界框

        Returns:
            裁剪后的人物图像或None
        """
        try:
            x1, y1, x2, y2 = bbox

            # 确保坐标有效
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            return frame[y1:y2, x1:x2]

        except Exception as e:
            logger.warning(f"裁剪人物图像失败: {e}")
            return None

    def process_video_directory(self, directory: str = None) -> Dict[str, Any]:
        """
        处理整个目录的视频文件

        Args:
            directory: 目录路径，None则使用配置中的默认目录

        Returns:
            所有视频的处理结果汇总
        """
        directory = directory or self.config.VIDEO_INPUT_DIR

        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        logger.info(f"开始处理目录: {directory}")

        # 获取视频文件列表
        video_files = get_video_files(directory, ['.mp4', '.avi', '.mov', '.mkv'])

        if not video_files:
            logger.warning(f"目录中没有找到视频文件: {directory}")
            return {'results': [], 'summary': {}}

        # 处理每个视频
        all_results = []
        successful_count = 0

        for video_file in video_files:
            try:
                result = self.process_single_video(video_file)

                if result is not None:
                    all_results.append({
                        'video_file': video_file,
                        'result': result
                    })
                    successful_count += 1

            except Exception as e:
                logger.error(f"处理视频失败 {video_file}: {e}")
                self.stats['errors'].append({
                    'video': video_file,
                    'error': str(e),
                    'timestamp': time.time()
                })

        # 创建汇总结果
        summary = {
            'total_videos': len(video_files),
            'successful_videos': successful_count,
            'failed_videos': len(video_files) - successful_count,
            'total_persons_across_videos': sum(
                len(item['result'].get('persons', [])) for item in all_results
            ),
            'processing_stats': self.get_processing_stats()
        }

        final_result = {
            'results': all_results,
            'summary': summary,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"目录处理完成: {successful_count}/{len(video_files)} 成功")

        return final_result

    def save_result(self, result: Dict[str, Any], output_path: str = None) -> bool:
        """
        保存处理结果

        Args:
            result: 处理结果
            output_path: 输出路径，None则自动生成

        Returns:
            是否成功保存
        """
        try:
            if output_path is None:
                # 自动生成输出路径
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    self.config.OUTPUT_DIR,
                    f'reid_results_{timestamp}.json'
                )

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存结果
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"结果保存成功: {output_path}")
            return True

        except Exception as e:
            logger.error(f"保存结果失败 {output_path}: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            统计信息字典
        """
        return self.stats.copy()

    def print_summary(self) -> None:
        """
        打印处理摘要
        """
        stats = self.get_processing_stats()

        print("\n" + "="*50)
        print("ReID处理摘要")
        print("="*50)
        print(f"处理视频数量: {stats['processed_videos']}")
        print(f"处理帧总数: {stats['total_frames_processed']}")
        print(f"检测人物总数: {stats['total_persons_detected']}")
        print(f"总处理时间: {stats['total_processing_time']:.2f}秒")

        if stats['total_frames_processed'] > 0:
            fps = stats['total_frames_processed'] / stats['total_processing_time']
            print(f"处理速度: {fps:.2f} 帧/秒")

        if stats['errors']:
            print(f"\n错误数量: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # 只显示前5个错误
                print(f"  - {error['video']}: {error['error']}")
            if len(stats['errors']) > 5:
                print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")

        print("="*50)

    def _cleanup(self) -> None:
        """
        清理资源
        """
        try:
            if self.video_processor:
                self.video_processor.close()

            # 其他清理操作可以在这里添加
            logger.info("资源清理完成")

        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

    def __del__(self):
        """析构函数"""
        self._cleanup()


def main():
    """
    测试函数
    """
    # 创建主控制器
    controller = MainController()

    try:
        # 初始化
        if controller.initialize():
            print("主控制器初始化成功")

            # 测试处理单个视频（需要实际的视频文件）
            test_video = "E:/ClaudeProject/ReID/video/test.mp4"

            if os.path.exists(test_video):
                print(f"\n处理测试视频: {test_video}")
                result = controller.process_single_video(test_video)

                if result:
                    print("处理成功！")
                    print(f"检测到 {len(result.get('persons', []))} 个人物")

                    # 保存结果
                    output_path = "E:/ClaudeProject/ReID/output/test_result.json"
                    if controller.save_result(result, output_path):
                        print(f"结果已保存到: {output_path}")
                else:
                    print("处理失败")
            else:
                print(f"测试视频不存在: {test_video}")

            # 打印摘要
            controller.print_summary()

        else:
            print("主控制器初始化失败")

    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()