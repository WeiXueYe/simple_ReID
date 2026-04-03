"""
出场时间分析模块
负责将离散的人物出现记录聚合成连续的时间段，并生成结构化结果
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

from .config import Config
from .utils import (
    frame_to_timestamp, timestamp_to_frame, format_duration,
    merge_time_segments, save_json, logger
)


class AppearanceAnalyzer:
    """
    出场时间分析器类
    将离散的出现记录聚合成连续的时间段，生成结构化结果
    """

    def __init__(self, config: Config = None):
        """
        初始化出场时间分析器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.video_info = {}

    def set_video_info(self, video_info: Dict[str, Any]) -> None:
        """
        设置视频信息

        Args:
            video_info: 视频信息字典
        """
        self.video_info = video_info.copy()
        logger.info(f"设置视频信息: {video_info['filename']}")

    def analyze_appearances(self, tracking_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        分析人物出场时间

        Args:
            tracking_results: 跟踪结果 {person_id: [appearance_records]}

        Returns:
            结构化分析结果
        """
        if not tracking_results:
            logger.warning("没有跟踪结果可供分析")
            return self._create_empty_result()

        try:
            # 分析每个人物的出场时间
            persons_data = []
            total_persons = len(tracking_results)
            total_appearances = 0

            for person_id, appearances in tracking_results.items():
                person_data = self._analyze_single_person(person_id, appearances)
                if person_data:
                    persons_data.append(person_data)
                    total_appearances += len(person_data['appearances'])

            # 创建最终结果
            result = {
                'video_info': self.video_info,
                'analysis_info': {
                    'total_persons': len(persons_data),
                    'total_appearances': total_appearances,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'config_used': {
                        'min_appearance_duration_ms': self.config.MIN_APPEARANCE_DURATION_MS,
                        'max_time_gap_ms': self.config.MAX_TIME_GAP_MS,
                        'frame_sample_rate': self.config.FRAME_SAMPLE_RATE
                    }
                },
                'persons': persons_data
            }

            logger.info(f"分析完成: {len(persons_data)} 个人物，{total_appearances} 个出场时间段")
            return result

        except Exception as e:
            logger.error(f"出场时间分析失败: {e}")
            return self._create_empty_result()

    def _analyze_single_person(self, person_id: str, appearances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        分析单个人物的出场时间

        Args:
            person_id: 人物ID
            appearances: 出现记录列表

        Returns:
            人物出场数据或None（如果过滤后为空）
        """
        if not appearances:
            return None

        try:
            # 提取帧号列表
            frame_numbers = []
            for appearance in appearances:
                frame_num = appearance.get('frame_number', 0)
                if isinstance(frame_num, (int, float)) and frame_num >= 0:
                    frame_numbers.append(int(frame_num))

            if not frame_numbers:
                return None

            # 聚合连续时间段
            time_segments = self._aggregate_time_segments(frame_numbers)

            # 过滤短时间出场
            filtered_segments = self._filter_short_appearances(time_segments)

            # 如果过滤后为空，返回None
            if not filtered_segments:
                logger.debug(f"人物 {person_id} 的所有出场时间段都被过滤")
                return None

            # 转换为最终格式
            appearances_data = []
            for start_frame, end_frame in filtered_segments:
                start_time_ms = frame_to_timestamp(start_frame, self.video_info.get('fps', 30))
                end_time_ms = frame_to_timestamp(end_frame, self.video_info.get('fps', 30))
                duration_ms = end_time_ms - start_time_ms

                appearance_data = {
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_ms': start_time_ms,
                    'end_time_ms': end_time_ms,
                    'duration_ms': duration_ms,
                    'start_timecode': self._frame_to_timecode(start_frame),
                    'end_timecode': self._frame_to_timecode(end_frame),
                    'duration_formatted': format_duration(duration_ms)
                }

                if self.config.INCLUDE_FRAME_DETAILS:
                    appearance_data['frame_count'] = end_frame - start_frame + 1

                appearances_data.append(appearance_data)

            # 计算总出场时间
            total_duration = sum(seg['duration_ms'] for seg in appearances_data)

            return {
                'person_id': person_id,
                'total_appearances': len(appearances_data),
                'total_duration_ms': total_duration,
                'total_duration_formatted': format_duration(total_duration),
                'first_appearance_frame': min(seg['start_frame'] for seg in appearances_data),
                'last_appearance_frame': max(seg['end_frame'] for seg in appearances_data),
                'appearances': appearances_data
            }

        except Exception as e:
            logger.error(f"分析人物 {person_id} 失败: {e}")
            return None

    def _aggregate_time_segments(self, frame_numbers: List[int]) -> List[Tuple[int, int]]:
        """
        将离散的帧号聚合成连续的时间段

        Args:
            frame_numbers: 帧号列表

        Returns:
            时间段列表 [(start_frame, end_frame), ...]
        """
        if not frame_numbers:
            return []

        # 排序并去重
        sorted_frames = sorted(set(frame_numbers))

        # 考虑采样率的影响
        max_gap_frames = max(1, self.config.MAX_TIME_GAP_MS * self.video_info.get('fps', 30) // 1000)

        segments = []
        start_frame = sorted_frames[0]
        prev_frame = start_frame

        for frame in sorted_frames[1:]:
            # 如果帧间隔超过阈值，开始新的时间段
            if frame - prev_frame > max_gap_frames:
                segments.append((start_frame, prev_frame))
                start_frame = frame

            prev_frame = frame

        # 添加最后一个时间段
        segments.append((start_frame, prev_frame))

        return segments

    def _filter_short_appearances(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        过滤短时间的出场

        Args:
            segments: 时间段列表

        Returns:
            过滤后的时间段列表
        """
        if not segments:
            return []

        min_duration_frames = max(1, self.config.MIN_APPEARANCE_DURATION_MS * self.video_info.get('fps', 30) // 1000)

        filtered = []
        for start_frame, end_frame in segments:
            duration_frames = end_frame - start_frame + 1
            if duration_frames >= min_duration_frames:
                filtered.append((start_frame, end_frame))
            else:
                logger.debug(f"过滤短时间出场: {start_frame}-{end_frame} ({duration_frames} 帧)")

        return filtered

    def _frame_to_timecode(self, frame_number: int) -> str:
        """
        将帧号转换为时间码

        Args:
            frame_number: 帧号

        Returns:
            时间码字符串
        """
        fps = self.video_info.get('fps', 30)
        return self._seconds_to_timecode(frame_number / fps)

    def _seconds_to_timecode(self, seconds: float) -> str:
        """
        将秒数转换为时间码

        Args:
            seconds: 秒数

        Returns:
            时间码字符串 (HH:MM:SS.mmm)
        """
        total_seconds = int(seconds)
        milliseconds = int((seconds - total_seconds) * 1000)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def validate_result(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证分析结果的有效性

        Args:
            result: 分析结果

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        try:
            # 检查必需字段
            required_fields = ['video_info', 'analysis_info', 'persons']
            for field in required_fields:
                if field not in result:
                    errors.append(f"缺少必需字段: {field}")

            # 检查视频信息
            if 'video_info' in result:
                video_info = result['video_info']
                if not isinstance(video_info, dict):
                    errors.append("video_info 必须是字典")
                elif 'fps' in video_info and video_info['fps'] <= 0:
                    errors.append("fps 必须大于0")

            # 检查人物数据
            if 'persons' in result:
                persons = result['persons']
                if not isinstance(persons, list):
                    errors.append("persons 必须是列表")
                else:
                    for i, person in enumerate(persons):
                        if not isinstance(person, dict):
                            errors.append(f"人物 {i} 数据格式错误")
                            continue

                        # 检查人物必需字段
                        person_required = ['person_id', 'appearances']
                        for field in person_required:
                            if field not in person:
                                errors.append(f"人物 {i} 缺少字段: {field}")

                        # 检查出场时间段
                        if 'appearances' in person:
                            appearances = person['appearances']
                            if not isinstance(appearances, list):
                                errors.append(f"人物 {i} 的 appearances 必须是列表")
                            else:
                                for j, appearance in enumerate(appearances):
                                    if not isinstance(appearance, dict):
                                        errors.append(f"人物 {i} 出场 {j} 数据格式错误")
                                        continue

                                    # 检查时间逻辑
                                    if 'start_frame' in appearance and 'end_frame' in appearance:
                                        if appearance['start_frame'] > appearance['end_frame']:
                                            errors.append(f"人物 {i} 出场 {j} 开始帧大于结束帧")

                                    if 'start_time_ms' in appearance and 'end_time_ms' in appearance:
                                        if appearance['start_time_ms'] > appearance['end_time_ms']:
                                            errors.append(f"人物 {i} 出场 {j} 开始时间大于结束时间")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"验证过程中发生错误: {e}")
            return False, errors

    def clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理和优化结果数据

        Args:
            result: 原始结果

        Returns:
            清理后的结果
        """
        try:
            cleaned = result.copy()

            # 清理人物数据
            if 'persons' in cleaned:
                cleaned_persons = []
                for person in cleaned['persons']:
                    # 过滤空的出场记录
                    if person.get('appearances'):
                        # 按开始时间排序出场记录
                        person['appearances'].sort(key=lambda x: x['start_frame'])
                        cleaned_persons.append(person)

                # 按人物ID排序
                cleaned_persons.sort(key=lambda x: x['person_id'])
                cleaned['persons'] = cleaned_persons

            # 更新统计信息
            if 'persons' in cleaned:
                cleaned['analysis_info']['total_persons'] = len(cleaned['persons'])
                cleaned['analysis_info']['total_appearances'] = sum(
                    len(p.get('appearances', [])) for p in cleaned['persons']
                )

            logger.info("结果清理完成")
            return cleaned

        except Exception as e:
            logger.error(f"结果清理失败: {e}")
            return result

    def save_result(self, result: Dict[str, Any], filepath: str) -> bool:
        """
        保存分析结果到文件

        Args:
            result: 分析结果
            filepath: 文件路径

        Returns:
            是否成功保存
        """
        try:
            # 验证结果
            is_valid, errors = self.validate_result(result)
            if not is_valid:
                logger.warning(f"结果验证失败: {errors}")

            # 清理结果
            cleaned_result = self.clean_result(result)

            # 保存文件
            save_json(cleaned_result, filepath)

            logger.info(f"结果保存成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存结果失败 {filepath}: {e}")
            return False

    def _create_empty_result(self) -> Dict[str, Any]:
        """
        创建空的结果结构

        Returns:
            空结果字典
        """
        return {
            'video_info': self.video_info,
            'analysis_info': {
                'total_persons': 0,
                'total_appearances': 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'error': 'No tracking results available'
            },
            'persons': []
        }

    def get_summary_statistics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取结果摘要统计

        Args:
            result: 分析结果

        Returns:
            统计信息
        """
        try:
            persons = result.get('persons', [])

            if not persons:
                return {
                    'total_persons': 0,
                    'total_appearances': 0,
                    'total_duration_ms': 0,
                    'avg_duration_ms': 0,
                    'min_duration_ms': 0,
                    'max_duration_ms': 0
                }

            # 计算统计信息
            total_appearances = sum(p.get('total_appearances', 0) for p in persons)
            total_duration = sum(p.get('total_duration_ms', 0) for p in persons)

            all_durations = []
            for person in persons:
                for appearance in person.get('appearances', []):
                    all_durations.append(appearance.get('duration_ms', 0))

            if all_durations:
                avg_duration = np.mean(all_durations)
                min_duration = np.min(all_durations)
                max_duration = np.max(all_durations)
            else:
                avg_duration = min_duration = max_duration = 0

            return {
                'total_persons': len(persons),
                'total_appearances': total_appearances,
                'total_duration_ms': total_duration,
                'total_duration_formatted': format_duration(total_duration),
                'avg_duration_ms': int(avg_duration),
                'min_duration_ms': int(min_duration),
                'max_duration_ms': int(max_duration),
                'avg_duration_formatted': format_duration(int(avg_duration))
            }

        except Exception as e:
            logger.error(f"计算统计信息失败: {e}")
            return {}


def main():
    """
    测试函数
    """
    # 创建分析器
    analyzer = AppearanceAnalyzer()

    # 模拟视频信息
    analyzer.set_video_info({
        'filename': 'test.mp4',
        'fps': 30,
        'frame_count': 900,
        'duration_seconds': 30
    })

    # 模拟跟踪结果
    test_tracking_results = {
        'person_001': [
            {'frame_number': 10},
            {'frame_number': 11},
            {'frame_number': 12},
            {'frame_number': 50},
            {'frame_number': 51}
        ],
        'person_002': [
            {'frame_number': 100},
            {'frame_number': 101},
            {'frame_number': 102}
        ]
    }

    try:
        # 分析出场时间
        result = analyzer.analyze_appearances(test_tracking_results)

        print("分析结果:")
        print(f"  总人物数: {result['analysis_info']['total_persons']}")
        print(f"  总出场次数: {result['analysis_info']['total_appearances']}")

        # 显示每个人物的信息
        for person in result['persons']:
            print(f"\n人物: {person['person_id']}")
            print(f"  出场次数: {person['total_appearances']}")
            print(f"  总时长: {person['total_duration_formatted']}")

            for i, appearance in enumerate(person['appearances']):
                print(f"    出场 {i+1}: {appearance['start_timecode']} - {appearance['end_timecode']}")

        # 验证结果
        is_valid, errors = analyzer.validate_result(result)
        print(f"\n结果验证: {'通过' if is_valid else '失败'}")
        if errors:
            print(f"  错误: {errors}")

        # 获取统计信息
        stats = analyzer.get_summary_statistics(result)
        print(f"\n统计信息: {stats}")

    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()