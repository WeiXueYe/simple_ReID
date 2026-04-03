"""
视频处理模块
负责视频文件的读取、帧提取和时间信息处理
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from .config import Config
from .utils import frame_to_timestamp, frame_to_timecode


# 配置日志
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    视频处理器类
    负责视频的读取、帧提取和基本信息获取
    """

    def __init__(self, config: Config = None):
        """
        初始化视频处理器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.cap = None
        self.video_info = {}

    def open_video(self, video_path: str) -> bool:
        """
        打开视频文件

        Args:
            video_path: 视频文件路径

        Returns:
            是否成功打开
        """
        try:
            # 关闭之前的视频（如果存在）
            if self.cap is not None:
                self.cap.release()

            # 打开新视频
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return False

            # 获取视频基本信息
            self._extract_video_info(video_path)

            logger.info(f"成功打开视频: {video_path}")
            logger.info(f"视频信息: {self.video_info}")

            return True

        except Exception as e:
            logger.error(f"打开视频文件失败 {video_path}: {e}")
            return False

    def _extract_video_info(self, video_path: str) -> None:
        """
        提取视频基本信息

        Args:
            video_path: 视频文件路径
        """
        try:
            # 基本信息
            self.video_info = {
                'filename': Path(video_path).name,
                'filepath': video_path,
                'file_size': Path(video_path).stat().st_size,
            }

            # 视频属性
            self.video_info.update({
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(self.cap.get(cv2.CAP_PROP_FPS)),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_seconds': None,
                'duration_ms': None,
            })

            # 计算时长
            if self.video_info['fps'] > 0:
                duration_seconds = self.video_info['frame_count'] / self.video_info['fps']
                self.video_info['duration_seconds'] = duration_seconds
                self.video_info['duration_ms'] = int(duration_seconds * 1000)

            # 编码格式
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.video_info['codec'] = self._fourcc_to_string(fourcc)

        except Exception as e:
            logger.error(f"提取视频信息失败: {e}")
            raise

    def _fourcc_to_string(self, fourcc: int) -> str:
        """
        将FOURCC编码转换为字符串

        Args:
            fourcc: FOURCC编码

        Returns:
            编码字符串
        """
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def get_video_info(self) -> Dict[str, Any]:
        """
        获取视频信息

        Returns:
            视频信息字典
        """
        return self.video_info.copy()

    def extract_frames(self, sample_rate: int = None) -> Iterator[Tuple[np.ndarray, int, int]]:
        """
        提取视频帧

        Args:
            sample_rate: 采样率，None则使用配置中的默认值

        Yields:
            (frame, frame_number, timestamp_ms)
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("视频未打开，请先调用open_video()")

        sample_rate = sample_rate or self.config.FRAME_SAMPLE_RATE

        logger.info(f"开始提取帧，采样率: {sample_rate}")

        frame_count = 0
        processed_count = 0

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    logger.info("视频读取完成")
                    break

                # 按采样率处理
                if frame_count % sample_rate == 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 计算时间戳
                    timestamp_ms = frame_to_timestamp(frame_count, self.video_info['fps'])

                    yield frame_rgb, frame_count, timestamp_ms
                    processed_count += 1

                frame_count += 1

                # 每100帧输出一次进度
                if frame_count % 100 == 0:
                    progress = (frame_count / self.video_info['frame_count']) * 100
                    logger.debug(f"帧提取进度: {progress:.1f}% ({frame_count}/{self.video_info['frame_count']})")

        except Exception as e:
            logger.error(f"帧提取过程中发生错误: {e}")
            raise

        finally:
            logger.info(f"帧提取完成，共处理 {processed_count} 帧")

    def extract_specific_frames(self, frame_numbers: list) -> Iterator[Tuple[np.ndarray, int, int]]:
        """
        提取指定帧号的帧

        Args:
            frame_numbers: 帧号列表

        Yields:
            (frame, frame_number, timestamp_ms)
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("视频未打开，请先调用open_video()")

        # 排序帧号
        sorted_frames = sorted(frame_numbers)

        logger.info(f"开始提取指定帧: {len(sorted_frames)} 帧")

        try:
            for frame_number in sorted_frames:
                # 设置帧位置
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"无法读取帧 {frame_number}")
                    continue

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 计算时间戳
                timestamp_ms = frame_to_timestamp(frame_number, self.video_info['fps'])

                yield frame_rgb, frame_number, timestamp_ms

        except Exception as e:
            logger.error(f"提取指定帧过程中发生错误: {e}")
            raise

    def get_frame_at_timestamp(self, timestamp_ms: int) -> Optional[Tuple[np.ndarray, int]]:
        """
        获取指定时间戳的帧

        Args:
            timestamp_ms: 时间戳（毫秒）

        Returns:
            (frame, frame_number) 或 None（如果失败）
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("视频未打开，请先调用open_video()")

        try:
            # 计算帧号
            frame_number = int((timestamp_ms / 1000) * self.video_info['fps'])

            # 限制帧号范围
            frame_number = max(0, min(frame_number, self.video_info['frame_count'] - 1))

            # 设置帧位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"无法读取时间戳 {timestamp_ms}ms 对应的帧")
                return None

            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame_rgb, frame_number

        except Exception as e:
            logger.error(f"获取指定时间戳帧失败: {e}")
            return None

    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """
        保存帧图像到文件

        Args:
            frame: 帧图像
            filepath: 保存路径

        Returns:
            是否成功保存
        """
        try:
            # 确保目录存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # 转换RGB到BGR用于保存
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            success = cv2.imwrite(filepath, frame_bgr)
            if success:
                logger.debug(f"帧保存成功: {filepath}")
            else:
                logger.error(f"帧保存失败: {filepath}")

            return success

        except Exception as e:
            logger.error(f"保存帧失败 {filepath}: {e}")
            return False

    def get_frame_statistics(self) -> Dict[str, Any]:
        """
        获取帧处理统计信息

        Returns:
            统计信息字典
        """
        if self.cap is None:
            return {}

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        return {
            'current_frame': current_frame,
            'total_frames': self.video_info.get('frame_count', 0),
            'progress_percentage': (current_frame / self.video_info.get('frame_count', 1)) * 100 if self.video_info.get('frame_count', 0) > 0 else 0,
            'current_time_ms': frame_to_timestamp(current_frame, self.video_info.get('fps', 30)),
            'current_timecode': frame_to_timecode(current_frame, self.video_info.get('fps', 30)),
        }

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        跳转到指定帧

        Args:
            frame_number: 目标帧号

        Returns:
            是否成功跳转
        """
        if self.cap is None or not self.cap.isOpened():
            return False

        try:
            # 限制帧号范围
            frame_number = max(0, min(frame_number, self.video_info['frame_count'] - 1))

            success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if success:
                logger.debug(f"跳转到帧 {frame_number}")
            else:
                logger.warning(f"跳转帧失败: {frame_number}")

            return success

        except Exception as e:
            logger.error(f"跳转帧失败: {e}")
            return False

    def seek_to_timestamp(self, timestamp_ms: int) -> bool:
        """
        跳转到指定时间戳

        Args:
            timestamp_ms: 时间戳（毫秒）

        Returns:
            是否成功跳转
        """
        frame_number = int((timestamp_ms / 1000) * self.video_info['fps'])
        return self.seek_to_frame(frame_number)

    def close(self) -> None:
        """
        关闭视频文件
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("视频文件已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __del__(self):
        """析构函数"""
        self.close()


def main():
    """
    测试函数
    """
    # 创建视频处理器
    processor = VideoProcessor()

    # 测试视频路径（这里需要替换为实际的视频文件）
    test_video = "test_video.mp4"

    if not Path(test_video).exists():
        logger.warning(f"测试视频不存在: {test_video}")
        return

    try:
        # 打开视频
        if processor.open_video(test_video):
            # 获取视频信息
            info = processor.get_video_info()
            print("视频信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # 提取前10帧
            print("\n提取前10帧:")
            frame_count = 0
            for frame, frame_num, timestamp in processor.extract_frames(sample_rate=1):
                if frame_count >= 10:
                    break

                print(f"  帧 {frame_num}: {timestamp}ms")
                frame_count += 1

    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()