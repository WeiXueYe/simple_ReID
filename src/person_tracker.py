"""
人物跟踪模块
基于特征匹配的人物跟踪和ID管理
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from collections import defaultdict, OrderedDict

from .config import Config
from .utils import cosine_similarity, generate_person_id, detect_shot_transition, predict_person_position, logger


class PersonTracker:
    """
    人物跟踪器类
    基于特征匹配维护人物ID，处理新人物出现和旧人物跟踪
    """

    def __init__(self, config: Config = None):
        """
        初始化人物跟踪器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.known_persons = {}  # {person_id: person_info}
        self.tracking_results = defaultdict(list)  # {person_id: [appearance_records]}
        self.frame_count = 0
        self.next_person_id = 1
        self.is_initialized = False

        # 镜头切换相关状态
        self.prev_frame = None
        self.is_post_transition = False
        self.transition_recovery_count = 0
        self.person_histories = defaultdict(list)  # 人物轨迹历史

    def initialize(self) -> bool:
        """
        初始化跟踪器

        Returns:
            是否成功初始化
        """
        try:
            self.reset()
            self.is_initialized = True
            logger.info("人物跟踪器初始化完成")
            return True

        except Exception as e:
            logger.error(f"人物跟踪器初始化失败: {e}")
            return False

    def reset(self) -> None:
        """
        重置跟踪器状态
        """
        self.known_persons.clear()
        self.tracking_results.clear()
        self.frame_count = 0
        self.next_person_id = 1
        logger.info("跟踪器状态已重置")

    def update_frame(self, frame_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        更新帧数据，进行人物跟踪

        Args:
            frame_data: 当前帧检测到的人物数据列表
                         [{'bbox': bbox, 'feature': feature, 'confidence': conf, ...}]

        Returns:
            带ID的人物数据列表
        """
        if not self.is_initialized:
            raise RuntimeError("跟踪器未初始化，请先调用initialize()")

        self.frame_count += 1
        current_frame = self.frame_count

        # 更新已知人物的丢失帧数
        self._update_missed_frames(current_frame)

        # 移除丢失太久的人物
        self._remove_lost_persons(current_frame)

        # 如果没有检测到人物，直接返回空列表
        if not frame_data:
            return []

        # 为当前帧的每个人物分配ID
        tracked_persons = []
        used_ids = set()

        # 按置信度排序，优先处理高置信度检测
        sorted_detections = sorted(frame_data, key=lambda x: x.get('confidence', 0), reverse=True)

        for detection in sorted_detections:
            person_info = self._assign_person_id(detection, current_frame)
            if person_info:
                tracked_persons.append(person_info)
                used_ids.add(person_info['person_id'])

        # 每20帧进行一次ID合并检查
        if current_frame % 20 == 0 and len(self.known_persons) > 3:
            self._check_and_merge_similar_persons()

        # 更新跟踪结果
        for person_info in tracked_persons:
            person_id = person_info['person_id']
            appearance_record = {
                'frame_number': current_frame,
                'timestamp_ms': person_info.get('timestamp_ms', 0),
                'bbox': person_info['bbox'],
                'confidence': person_info['confidence'],
                'feature': person_info['feature'].copy() if person_info['feature'] is not None else None
            }
            self.tracking_results[person_id].append(appearance_record)

        logger.debug(f"帧 {current_frame}: 跟踪到 {len(tracked_persons)} 个人物")
        return tracked_persons

    def _assign_person_id(self, detection: Dict[str, Any], frame_number: int) -> Optional[Dict[str, Any]]:
        """
        为检测到的人物分配ID（增强版本）

        Args:
            detection: 检测到的人物数据
            frame_number: 当前帧号

        Returns:
            带ID的人物信息或None
        """
        feature = detection.get('feature')
        if feature is None:
            return None

        # 获取当前相似度阈值（考虑镜头切换状态）
        current_threshold = self._get_current_similarity_threshold()

        # 在已知人物中寻找最佳匹配
        best_match_id, best_similarity = self._find_best_match(
            feature,
            threshold=current_threshold
        )

        # 如果找到匹配且相似度足够高，使用现有ID
        if best_match_id and best_similarity >= current_threshold:
            person_info = self.known_persons[best_match_id]

            # 获取当前特征更新权重
            feature_weight = self._get_current_feature_weight()

            # 更新特征（特征融合）
            updated_feature = self._update_person_feature(
                person_info['feature'],
                feature,
                feature_weight
            )

            # 更新人物信息和历史
            person_info.update({
                'feature': updated_feature,
                'last_seen_frame': frame_number,
                'missed_frames': 0,
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0)
            })

            # 更新轨迹历史
            self._update_person_history(best_match_id, detection['bbox'], frame_number)

            return {
                'person_id': best_match_id,
                'bbox': detection['bbox'],
                'feature': updated_feature,
                'confidence': detection.get('confidence', 0),
                'frame_number': frame_number,
                'timestamp_ms': detection.get('timestamp_ms', 0),
                'similarity': best_similarity
            }

        # 否则创建新人物
        else:
            new_person_id = self._generate_new_person_id()

            # 创建新人物信息
            person_info = {
                'person_id': new_person_id,
                'feature': feature.copy(),
                'first_seen_frame': frame_number,
                'last_seen_frame': frame_number,
                'missed_frames': 0,
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0),
                'appearance_count': 1
            }

            self.known_persons[new_person_id] = person_info
            self._update_person_history(new_person_id, detection['bbox'], frame_number)

            logger.info(f"新人物出现: {new_person_id} (帧 {frame_number}, 相似度: {best_similarity:.3f})")

            return {
                'person_id': new_person_id,
                'bbox': detection['bbox'],
                'feature': feature.copy(),
                'confidence': detection.get('confidence', 0),
                'frame_number': frame_number,
                'timestamp_ms': detection.get('timestamp_ms', 0),
                'similarity': best_similarity
            }

    def _find_best_match(self, query_feature: np.ndarray, threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        在已知人物中找到最佳匹配
        增强版本：考虑时间连续性和空间位置

        Args:
            query_feature: 查询特征
            threshold: 相似度阈值

        Returns:
            (最佳匹配ID, 相似度分数)
        """
        if not self.known_persons:
            return None, 0.0

        candidates = []

        for person_id, person_info in self.known_persons.items():
            if person_info['feature'] is None:
                continue

            # 基础特征相似度
            feature_similarity = cosine_similarity(query_feature, person_info['feature'])

            # 如果基础相似度不够，跳过
            if feature_similarity < threshold * 0.8:  # 稍微放宽初步筛选
                continue

            # 时间连续性加分：最近出现的人物更有可能是匹配的
            time_bonus = 0.0
            if person_info['missed_frames'] <= 2:
                time_bonus = 0.05  # 最近出现的人物加分
            elif person_info['missed_frames'] <= 5:
                time_bonus = 0.02

            # 综合评分
            final_similarity = feature_similarity + time_bonus

            candidates.append((person_id, final_similarity, feature_similarity))

        if not candidates:
            return None, 0.0

        # 按综合评分排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, final_similarity, original_similarity = candidates[0]

        # 只有原始相似度达到阈值才认为是有效匹配
        if original_similarity >= threshold:
            return best_id, final_similarity
        else:
            return None, 0.0

    def _update_person_feature(self, existing_feature: np.ndarray,
                             new_feature: np.ndarray,
                             weight: float = 0.3) -> np.ndarray:
        """
        更新人物特征（特征融合）

        Args:
            existing_feature: 现有特征
            new_feature: 新特征
            weight: 新特征权重

        Returns:
            更新后的特征
        """
        if existing_feature is None:
            return new_feature.copy()

        # 线性插值融合特征
        updated_feature = (1 - weight) * existing_feature + weight * new_feature

        # 归一化
        norm = np.linalg.norm(updated_feature)
        if norm > 0:
            updated_feature = updated_feature / norm

        return updated_feature

    def _update_missed_frames(self, current_frame: int) -> None:
        """
        更新已知人物的丢失帧数

        Args:
            current_frame: 当前帧号
        """
        for person_id, person_info in self.known_persons.items():
            if person_info['last_seen_frame'] < current_frame:
                person_info['missed_frames'] = (
                    current_frame - person_info['last_seen_frame']
                )

    def _remove_lost_persons(self, current_frame: int) -> None:
        """
        移除丢失太久的人物

        Args:
            current_frame: 当前帧号
        """
        lost_persons = []

        for person_id, person_info in self.known_persons.items():
            if person_info['missed_frames'] > self.config.MAX_MISSED_FRAMES:
                lost_persons.append(person_id)

        for person_id in lost_persons:
            person_info = self.known_persons[person_id]
            logger.info(
                f"人物离开: {person_id} "
                f"(最后出现: 帧 {person_info['last_seen_frame']}, "
                f"丢失帧数: {person_info['missed_frames']})"
            )
            del self.known_persons[person_id]

    def _generate_new_person_id(self) -> str:
        """
        生成新的人物ID

        Returns:
            新的人物ID
        """
        person_id = f"person_{self.next_person_id:03d}"
        self.next_person_id += 1
        return person_id

    def update_empty_frame(self, frame_number: int) -> None:
        """
        更新空帧（没有检测到人物）

        Args:
            frame_number: 帧号
        """
        self.frame_count = frame_number
        self._update_missed_frames(frame_number)
        self._remove_lost_persons(frame_number)

        logger.debug(f"空帧 {frame_number}: 更新丢失帧数")

    def get_tracking_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取跟踪结果

        Returns:
            跟踪结果字典 {person_id: [appearance_records]}
        """
        return dict(self.tracking_results)

    def get_current_persons(self) -> Dict[str, Any]:
        """
        获取当前在场的人物

        Returns:
            当前人物信息字典
        """
        current_persons = {}

        for person_id, person_info in self.known_persons.items():
            if person_info['missed_frames'] == 0:
                current_persons[person_id] = person_info.copy()

        return current_persons

    def get_person_statistics(self) -> Dict[str, Any]:
        """
        获取跟踪统计信息

        Returns:
            统计信息字典
        """
        total_persons = len(self.known_persons)
        current_persons = len(self.get_current_persons())
        total_appearances = sum(len(records) for records in self.tracking_results.values())

        # 计算平均出场次数
        if total_persons > 0:
            avg_appearances = total_appearances / total_persons
        else:
            avg_appearances = 0

        # 计算最长跟踪时间
        max_tracking_frames = 0
        for person_info in self.known_persons.values():
            tracking_duration = (
                person_info['last_seen_frame'] - person_info['first_seen_frame'] + 1
            )
            max_tracking_frames = max(max_tracking_frames, tracking_duration)

        return {
            'total_persons_detected': total_persons,
            'current_persons_present': current_persons,
            'total_appearances': total_appearances,
            'average_appearances_per_person': round(avg_appearances, 2),
            'max_tracking_duration_frames': max_tracking_frames,
            'current_frame': self.frame_count,
            'next_person_id': self.next_person_id
        }

    def get_person_history(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定人物的历史信息

        Args:
            person_id: 人物ID

        Returns:
            人物历史信息或None
        """
        if person_id not in self.known_persons:
            return None

        person_info = self.known_persons[person_id]
        appearances = self.tracking_results.get(person_id, [])

        return {
            'person_id': person_id,
            'first_seen_frame': person_info['first_seen_frame'],
            'last_seen_frame': person_info['last_seen_frame'],
            'missed_frames': person_info['missed_frames'],
            'total_appearances': len(appearances),
            'is_currently_present': person_info['missed_frames'] == 0,
            'appearances': appearances.copy()
        }

    def find_similar_persons(self, feature: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        查找与给定特征相似的人物

        Args:
            feature: 查询特征
            threshold: 相似度阈值

        Returns:
            相似人物列表 [(person_id, similarity), ...]
        """
        similar_persons = []

        for person_id, person_info in self.known_persons.items():
            if person_info['feature'] is None:
                continue

            similarity = cosine_similarity(feature, person_info['feature'])
            if similarity >= threshold:
                similar_persons.append((person_id, similarity))

        # 按相似度降序排序
        similar_persons.sort(key=lambda x: x[1], reverse=True)

        return similar_persons

    def merge_persons(self, person_id1: str, person_id2: str) -> bool:
        """
        合并两个相似的人物（处理ID切换问题）

        Args:
            person_id1: 第一个人物ID
            person_id2: 第二个人物ID

        Returns:
            是否成功合并
        """
        if person_id1 not in self.known_persons or person_id2 not in self.known_persons:
            return False

        try:
            # 选择保留ID较小的那个
            keep_id = min(person_id1, person_id2)
            remove_id = max(person_id1, person_id2)

            # 合并跟踪记录
            records1 = self.tracking_results[keep_id]
            records2 = self.tracking_results[remove_id]
            merged_records = records1 + records2

            # 按帧号排序
            merged_records.sort(key=lambda x: x['frame_number'])

            # 更新跟踪结果
            self.tracking_results[keep_id] = merged_records
            del self.tracking_results[remove_id]

            # 更新已知人物信息
            person1_info = self.known_persons[keep_id]
            person2_info = self.known_persons[remove_id]

            # 合并特征（简单平均）
            if person1_info['feature'] is not None and person2_info['feature'] is not None:
                merged_feature = (person1_info['feature'] + person2_info['feature']) / 2
                merged_feature = merged_feature / np.linalg.norm(merged_feature)
                person1_info['feature'] = merged_feature

            # 更新时间范围
            person1_info['first_seen_frame'] = min(
                person1_info['first_seen_frame'],
                person2_info['first_seen_frame']
            )
            person1_info['last_seen_frame'] = max(
                person1_info['last_seen_frame'],
                person2_info['last_seen_frame']
            )

            # 移除被合并的人物
            del self.known_persons[remove_id]

            logger.info(f"合并人物: {remove_id} -> {keep_id}")
            return True

        except Exception as e:
            logger.error(f"合并人物失败 {person_id1} 和 {person_id2}: {e}")
            return False

    def _check_and_merge_similar_persons(self) -> None:
        """
        检查并合并相似的人物ID
        """
        if len(self.known_persons) < 2:
            return

        person_ids = list(self.known_persons.keys())
        merged_pairs = []

        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id1, id2 = person_ids[i], person_ids[j]

                # 跳过已经被合并的
                if id1 not in self.known_persons or id2 not in self.known_persons:
                    continue

                person1 = self.known_persons[id1]
                person2 = self.known_persons[id2]

                # 如果两个特征都存在，计算相似度
                if person1['feature'] is not None and person2['feature'] is not None:
                    similarity = cosine_similarity(person1['feature'], person2['feature'])

                    # 如果相似度很高，考虑合并
                    if similarity >= 0.85:  # 合并阈值
                        # 检查时间重叠情况
                        overlap = self._check_temporal_overlap(id1, id2)
                        if overlap:
                            merged_pairs.append((id1, id2, similarity))

        # 执行合并
        for id1, id2, similarity in merged_pairs:
            if id1 in self.known_persons and id2 in self.known_persons:
                self.merge_persons(id1, id2)

    def _check_temporal_overlap(self, id1: str, id2: str) -> bool:
        """
        检查两个ID是否有时间重叠
        """
        records1 = self.tracking_results.get(id1, [])
        records2 = self.tracking_results.get(id2, [])

        if not records1 or not records2:
            return False

        # 获取时间范围
        frames1 = {r['frame_number'] for r in records1}
        frames2 = {r['frame_number'] for r in records2}

        # 检查是否有重叠或接近的帧
        overlap_threshold = 10  # 10帧内的间隔认为是相关的
        for f1 in frames1:
            for f2 in frames2:
                if abs(f1 - f2) <= overlap_threshold:
                    return True

        return False

    def _get_current_similarity_threshold(self) -> float:
        """
        获取当前相似度阈值（考虑镜头切换状态）

        Returns:
            当前阈值
        """
        if self.is_post_transition:
            # 镜头切换后使用更宽松的阈值
            return self.config.POST_TRANSITION_SIMILARITY_THRESHOLD
        else:
            # 正常情况使用标准阈值
            return self.config.FEATURE_SIMILARITY_THRESHOLD

    def _get_current_feature_weight(self) -> float:
        """
        获取当前特征更新权重（考虑镜头切换状态）

        Returns:
            当前权重
        """
        if self.is_post_transition:
            # 镜头切换后使用更高的更新权重
            return self.config.POST_TRANSITION_FEATURE_WEIGHT
        else:
            # 正常情况使用标准权重
            return self.config.FEATURE_UPDATE_WEIGHT

    def _get_current_missed_frames_limit(self) -> int:
        """
        获取当前丢失帧数限制（考虑镜头切换状态）

        Returns:
            当前限制
        """
        if self.is_post_transition:
            # 镜头切换后容忍更多丢失帧
            return self.config.POST_TRANSITION_MISSED_FRAMES
        else:
            # 正常情况使用标准限制
            return self.config.MAX_MISSED_FRAMES

    def _update_person_history(self, person_id: str, bbox: Tuple[int, int, int, int], frame_number: int) -> None:
        """
        更新人物轨迹历史

        Args:
            person_id: 人物ID
            bbox: 边界框
            frame_number: 帧号
        """
        history = self.person_histories[person_id]
        history.append({
            'bbox': bbox,
            'frame': frame_number
        })

        # 保持最近50帧的历史
        if len(history) > 50:
            history.pop(0)

    def detect_and_handle_shot_transition(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """
        检测并处理镜头切换

        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧

        Returns:
            是否检测到镜头切换
        """
        if not self.config.ENABLE_SHOT_TRANSITION_DETECTION:
            return False

        is_transition = detect_shot_transition(
            prev_frame, curr_frame, self.config.SHOT_TRANSITION_THRESHOLD
        )

        if is_transition:
            logger.info(f"检测到镜头切换 (帧 {self.frame_count})")
            self.is_post_transition = True
            self.transition_recovery_count = 0

            # 在切换时执行紧急ID合并
            self._emergency_merge_similar_persons()

        # 更新恢复计数
        if self.is_post_transition:
            self.transition_recovery_count += 1
            if self.transition_recovery_count >= self.config.TRANSITION_RECOVERY_FRAMES:
                self.is_post_transition = False
                logger.debug(f"镜头切换恢复完成 (帧 {self.frame_count})")

        return is_transition

    def _emergency_merge_similar_persons(self) -> None:
        """
        紧急合并相似人物（镜头切换时）
        """
        if len(self.known_persons) < 2:
            return

        person_ids = list(self.known_persons.keys())
        merged_pairs = []

        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id1, id2 = person_ids[i], person_ids[j]

                if id1 not in self.known_persons or id2 not in self.known_persons:
                    continue

                person1 = self.known_persons[id1]
                person2 = self.known_persons[id2]

                if person1['feature'] is not None and person2['feature'] is not None:
                    similarity = cosine_similarity(person1['feature'], person2['feature'])

                    # 镜头切换时使用更宽松的合并阈值
                    if similarity >= 0.70:  # 比正常的0.85更宽松
                        merged_pairs.append((id1, id2, similarity))

        # 执行合并
        for id1, id2, similarity in merged_pairs:
            if id1 in self.known_persons and id2 in self.known_persons:
                logger.info(f"镜头切换紧急合并: {id1} + {id2} (相似度: {similarity:.3f})")
                self.merge_persons(id1, id2)

    def _remove_lost_persons(self, current_frame: int) -> None:
        """
        移除丢失太久的人物（增强版本）

        Args:
            current_frame: 当前帧号
        """
        current_limit = self._get_current_missed_frames_limit()
        lost_persons = []

        for person_id, person_info in self.known_persons.items():
            if person_info['missed_frames'] > current_limit:
                lost_persons.append(person_id)

        for person_id in lost_persons:
            person_info = self.known_persons[person_id]
            logger.info(
                f"人物离开: {person_id} "
                f"(最后出现: 帧 {person_info['last_seen_frame']}, "
                f"丢失帧数: {person_info['missed_frames']})"
            )
            del self.known_persons[person_id]
            # 清理历史
            if person_id in self.person_histories:
                del self.person_histories[person_id]

    def __len__(self) -> int:
        """
        返回当前跟踪的人物数量
        """
        return len(self.known_persons)


def main():
    """
    测试函数
    """
    # 创建跟踪器
    tracker = PersonTracker()

    try:
        # 初始化
        if tracker.initialize():
            print("跟踪器初始化成功")

            # 模拟一些测试数据
            test_features = []
            for i in range(5):
                # 生成随机特征（实际使用中这些来自特征提取器）
                feature = np.random.randn(512).astype(np.float32)
                feature = feature / np.linalg.norm(feature)
                test_features.append(feature)

            # 模拟几帧数据
            for frame in range(1, 11):
                print(f"\n处理帧 {frame}:")

                # 随机选择一些特征作为检测结果
                num_detections = np.random.randint(0, 4)
                frame_data = []

                for i in range(num_detections):
                    feature_idx = np.random.randint(0, len(test_features))
                    detection = {
                        'feature': test_features[feature_idx],
                        'bbox': (100, 100, 200, 300),
                        'confidence': 0.8 + np.random.random() * 0.2,
                        'frame_number': frame,
                        'timestamp_ms': frame * 33  # 假设30fps
                    }
                    frame_data.append(detection)

                # 更新跟踪器
                tracked_persons = tracker.update_frame(frame_data)

                print(f"  检测到 {num_detections} 个人物，跟踪到 {len(tracked_persons)} 个")

                for person in tracked_persons:
                    print(f"    - {person['person_id']}: 置信度 {person['confidence']:.2f}")

                # 显示统计信息
                if frame % 5 == 0:
                    stats = tracker.get_person_statistics()
                    print(f"  统计: {stats['current_persons_present']} 人在场, "
                          f"{stats['total_persons_detected']} 人总计")

            # 显示最终结果
            print("\n最终跟踪结果:")
            results = tracker.get_tracking_results()
            for person_id, appearances in results.items():
                print(f"  {person_id}: {len(appearances)} 次出场")

        else:
            print("跟踪器初始化失败")

    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()