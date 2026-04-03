"""
项目配置文件
包含所有模块的配置参数
"""

import os
from pathlib import Path


class Config:
    """
    系统配置类
    集中管理所有配置参数，便于维护和修改
    """

    # ==================== 路径配置 ====================
    # 视频输入目录
    VIDEO_INPUT_DIR = "E:/ClaudeProject/ReID/video"

    # 输出结果目录
    OUTPUT_DIR = "E:/ClaudeProject/ReID/output"

    # 模型文件目录
    MODEL_DIR = "E:/ClaudeProject/ReID/models"

    # ==================== 视频处理配置 ====================
    # 帧采样率：1表示处理每一帧，2表示每隔一帧处理，以此类推
    # 值越大处理速度越快，但精度可能降低
    FRAME_SAMPLE_RATE = 1

    # 视频读取缓冲区大小（MB）
    VIDEO_BUFFER_SIZE = 32

    # ==================== 人物检测配置（YOLOv8）====================
    # 检测置信度阈值：低于此值的检测结果将被过滤
    # 范围：0.0-1.0，值越高检测越严格，假阳性越少
    DETECTION_CONFIDENCE_THRESHOLD = 0.05

    # YOLO模型版本选择：'n'（nano）、's'（small）、'm'（medium）、'l'（large）、'x'（xlarge）
    # 模型越大精度越高，但处理速度越慢
    YOLO_MODEL_SIZE = 's'  # 使用small版本，平衡精度和速度

    # 检测类别：0表示只检测人物
    DETECT_CLASSES = [0]

    # 输入图像大小：YOLO模型输入尺寸
    # 常见值：416, 640, 1280，值越大精度越高但速度越慢
    YOLO_IMG_SIZE = 1280  # 提高输入分辨率以提升检测精度

    # ==================== 特征提取配置 ====================
    # 人物ReID模型选择
    # 可选：'osnet_x1_0', 'pcb_r50', 'mlfn', 'resnet50'
    REID_MODEL_NAME = 'se_resnext50_32x4d'  # 更精确的模型，适合复杂场景

    # 特征向量维度
    FEATURE_DIM = 512

    # 特征相似度阈值：高于此值认为是同一个人
    # 范围：0.0-1.0，值越高匹配越严格
    FEATURE_SIMILARITY_THRESHOLD = 0.65

    # 特征提取图像尺寸
    REID_IMG_WIDTH = 256  # 提高输入分辨率以提升特征质量
    REID_IMG_HEIGHT = 512

    # ==================== 人物跟踪配置 ====================
    # 最大丢失帧数：人物消失多少帧后认为其离开
    # 值越大跟踪越稳定，但可能导致ID切换延迟
    MAX_MISSED_FRAMES = 25

    # 新人物创建阈值：低于此相似度则创建新人物ID
    # 应小于 FEATURE_SIMILARITY_THRESHOLD
    NEW_PERSON_THRESHOLD = 0.6

    # 特征更新权重：新特征与历史特征的融合权重
    # 0.0表示只使用历史特征，1.0表示只使用新特征
    FEATURE_UPDATE_WEIGHT = 0.15

    # ==================== 出场时间分析配置 ====================
    # 最小出场时长（毫秒）：短于此时间段的出场将被过滤
    # 用于去除误检和短暂噪声
    MIN_APPEARANCE_DURATION_MS = 500

    # 最大时间间隔（毫秒）：间隔小于此值的时间段将被合并
    # 用于处理因遮挡导致的短暂消失
    MAX_TIME_GAP_MS = 1000

    # ==================== 镜头切换处理配置 ====================
    # 镜头切换检测
    ENABLE_SHOT_TRANSITION_DETECTION = True
    SHOT_TRANSITION_THRESHOLD = 0.3  # 帧间差异阈值

    # 动态相似度阈值
    POST_TRANSITION_SIMILARITY_THRESHOLD = 0.58  # 切换后阈值
    TRANSITION_RECOVERY_FRAMES = 30  # 恢复到正常阈值的帧数

    # 切换后处理
    POST_TRANSITION_MISSED_FRAMES = 15  # 切换后丢失帧数容忍度
    POST_TRANSITION_FEATURE_WEIGHT = 0.35  # 切换后特征更新权重

    # ==================== 性能配置 ====================
    # 批处理大小：特征提取时的批处理大小
    # 值越大内存使用越高，但处理速度可能更快
    # GPU用户建议设置为16-32，CPU用户建议设置为4-8
    BATCH_SIZE = 16

    # 是否使用GPU加速
    USE_GPU = True

    # GPU设备ID，-1表示使用CPU
    GPU_DEVICE = 0

    # 特征缓存大小：缓存最近处理的特征数量
    FEATURE_CACHE_SIZE = 1000

    # ==================== 日志和调试配置 ====================
    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = 'INFO'

    # 是否保存中间结果（调试用）
    SAVE_INTERMEDIATE_RESULTS = False

    # 中间结果保存目录
    DEBUG_OUTPUT_DIR = "E:/ClaudeProject/ReID/debug"

    # ==================== 输出配置 ====================
    # 输出文件格式：'json', 'csv', 'txt'
    OUTPUT_FORMAT = 'json'

    # 是否包含详细的帧级信息
    INCLUDE_FRAME_DETAILS = False

    # 结果文件命名格式
    RESULT_FILENAME_FORMAT = "{video_name}_reid_result.{ext}"

    @classmethod
    def validate_config(cls):
        """
        验证配置参数的有效性
        """
        # 检查路径是否存在，不存在则创建
        for path_attr in ['VIDEO_INPUT_DIR', 'OUTPUT_DIR', 'MODEL_DIR']:
            path = getattr(cls, path_attr)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"创建目录: {path}")

        # 验证阈值范围
        assert 0.0 <= cls.DETECTION_CONFIDENCE_THRESHOLD <= 1.0, "检测置信度阈值必须在0-1之间"
        assert 0.0 <= cls.FEATURE_SIMILARITY_THRESHOLD <= 1.0, "特征相似度阈值必须在0-1之间"
        assert 0.0 <= cls.NEW_PERSON_THRESHOLD <= 1.0, "新人物阈值必须在0-1之间"
        assert 0.0 <= cls.FEATURE_UPDATE_WEIGHT <= 1.0, "特征更新权重必须在0-1之间"

        # 验证数值范围
        assert cls.FRAME_SAMPLE_RATE >= 1, "帧采样率必须大于等于1"
        assert cls.MAX_MISSED_FRAMES >= 0, "最大丢失帧数必须大于等于0"
        assert cls.MIN_APPEARANCE_DURATION_MS >= 0, "最小出场时长必须大于等于0"
        assert cls.MAX_TIME_GAP_MS >= 0, "最大时间间隔必须大于等于0"
        assert cls.BATCH_SIZE >= 1, "批处理大小必须大于等于1"

        print("配置验证通过")

    @classmethod
    def get_yolo_model_path(cls):
        """
        获取YOLO模型路径
        """
        return os.path.join(cls.MODEL_DIR, f'yolov8{cls.YOLO_MODEL_SIZE}.pt')

    @classmethod
    def get_reid_model_path(cls):
        """
        获取ReID模型路径
        """
        return os.path.join(cls.MODEL_DIR, f'{cls.REID_MODEL_NAME}.pth')

    @classmethod
    def get_output_path(cls, video_filename):
        """
        获取输出文件路径
        """
        video_name = Path(video_filename).stem
        filename = cls.RESULT_FILENAME_FORMAT.format(
            video_name=video_name,
            ext=cls.OUTPUT_FORMAT
        )
        return os.path.join(cls.OUTPUT_DIR, filename)


# 创建全局配置实例
config = Config()

# 导入时自动验证配置
config.validate_config()