"""
主程序入口
视频人物ReID系统的主入口点
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.main_controller import MainController
from src.utils import get_video_files, logger


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    # 创建日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置根日志
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format
    )

    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        # 添加到根日志
        logging.getLogger().addHandler(file_handler)

        logger.info(f"日志将同时输出到文件: {log_file}")


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='视频人物ReID系统 - 自动提取视频中的人物出场时间',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 处理默认目录中的所有视频
  python main.py -i ./videos                       # 处理指定目录中的视频
  python main.py -o ./results                       # 指定输出目录
  python main.py --log-level DEBUG                  # 设置调试日志级别
  python main.py --log-file ./logs/reid.log         # 指定日志文件
  python main.py --config custom_config.json        # 使用自定义配置文件
  python main.py --single video.mp4                 # 处理单个视频文件
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default=None,
        help='输入视频目录路径 (默认: 配置文件中的VIDEO_INPUT_DIR)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='输出结果目录路径 (默认: 配置文件中的OUTPUT_DIR)'
    )

    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='处理单个视频文件'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='日志文件路径 (默认: 只输出到控制台)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径'
    )

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv'],
        help='支持的视频文件扩展名 (默认: .mp4 .avi .mov .mkv)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用GPU加速，强制使用CPU'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=None,
        help='帧采样率 (1=处理每一帧, 2=每隔一帧处理, 默认: 配置文件中的设置)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=None,
        help='检测置信度阈值 (0.0-1.0, 默认: 配置文件中的设置)'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    验证命令行参数

    Args:
        args: 解析后的参数

    Returns:
        参数是否有效
    """
    # 检查输入路径
    if args.single:
        if not os.path.exists(args.single):
            print(f"错误: 视频文件不存在: {args.single}")
            return False
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"错误: 输入目录不存在: {args.input_dir}")
            return False

    # 检查置信度阈值
    if args.confidence_threshold is not None:
        if not 0.0 <= args.confidence_threshold <= 1.0:
            print("错误: 置信度阈值必须在0.0到1.0之间")
            return False

    # 检查采样率
    if args.sample_rate is not None:
        if args.sample_rate < 1:
            print("错误: 采样率必须大于等于1")
            return False

    return True


def print_banner() -> None:
    """
    打印程序横幅
    """
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    视频人物ReID系统                         ║
    ║              Video Person Re-Identification                 ║
    ║                                                              ║
    ║        自动提取视频中的人物出场时间信息                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_configuration(config: Config, args: argparse.Namespace) -> None:
    """
    打印当前配置

    Args:
        config: 配置对象
        args: 命令行参数
    """
    print("\n当前配置:")
    print("=" * 50)
    print(f"输入目录: {args.input_dir or config.VIDEO_INPUT_DIR}")
    print(f"输出目录: {args.output_dir or config.OUTPUT_DIR}")
    print(f"日志级别: {args.log_level}")

    if args.single:
        print(f"处理模式: 单个文件 ({args.single})")
    else:
        print(f"处理模式: 批量处理")

    print(f"帧采样率: {args.sample_rate or config.FRAME_SAMPLE_RATE}")
    print(f"置信度阈值: {args.confidence_threshold or config.DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"使用GPU: {not args.no_gpu and config.USE_GPU}")
    print(f"支持的文件类型: {', '.join(args.extensions)}")
    print("=" * 50)


def update_config_from_args(config: Config, args: argparse.Namespace) -> None:
    """
    根据命令行参数更新配置

    Args:
        config: 配置对象
        args: 命令行参数
    """
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    if args.no_gpu:
        config.USE_GPU = False

    if args.sample_rate:
        config.FRAME_SAMPLE_RATE = args.sample_rate

    if args.confidence_threshold:
        config.DETECTION_CONFIDENCE_THRESHOLD = args.confidence_threshold


def process_single_video_file(controller: MainController, video_path: str) -> Dict[str, Any]:
    """
    处理单个视频文件

    Args:
        controller: 主控制器
        video_path: 视频文件路径

    Returns:
        处理结果
    """
    print(f"\n处理视频: {os.path.basename(video_path)}")
    print("-" * 50)

    start_time = time.time()

    try:
        result = controller.process_single_video(video_path)

        if result:
            processing_time = time.time() - start_time
            persons_count = len(result.get('persons', []))

            print(f"✓ 处理成功 (耗时: {processing_time:.2f}秒)")
            print(f"  检测到人物: {persons_count} 个")

            # 显示每个人物的简要信息
            for person in result.get('persons', []):
                appearances = len(person.get('appearances', []))
                duration = person.get('total_duration_formatted', '未知')
                print(f"    - {person['person_id']}: {appearances} 次出场, 总时长 {duration}")

            return result
        else:
            print("✗ 处理失败")
            return {}

    except Exception as e:
        print(f"✗ 处理失败: {e}")
        return {}


def process_video_directory(controller: MainController, directory: str, extensions: List[str]) -> Dict[str, Any]:
    """
    处理视频目录

    Args:
        controller: 主控制器
        directory: 目录路径
        extensions: 支持的文件扩展名

    Returns:
        处理结果汇总
    """
    print(f"\n扫描目录: {directory}")
    print("-" * 50)

    # 获取视频文件列表
    video_files = get_video_files(directory, extensions)

    if not video_files:
        print(f"在目录中未找到支持的视频文件: {directory}")
        print(f"支持的文件类型: {', '.join(extensions)}")
        return {'results': [], 'summary': {}}

    print(f"找到 {len(video_files)} 个视频文件:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")

    # 处理每个视频
    all_results = []
    successful_count = 0
    failed_count = 0

    print(f"\n开始处理视频...")
    print("=" * 50)

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {os.path.basename(video_file)}")

        try:
            result = process_single_video_file(controller, video_file)

            if result:
                all_results.append({
                    'video_file': video_file,
                    'result': result
                })
                successful_count += 1

                # 保存单个视频的结果
                output_filename = f"{Path(video_file).stem}_reid_result.json"
                output_path = os.path.join(controller.config.OUTPUT_DIR, output_filename)
                controller.save_result(result, output_path)

            else:
                failed_count += 1

        except Exception as e:
            print(f"✗ 处理失败: {e}")
            failed_count += 1

    # 创建汇总结果
    summary = {
        'total_videos': len(video_files),
        'successful_videos': successful_count,
        'failed_videos': failed_count,
        'total_persons_across_videos': sum(
            len(item['result'].get('persons', [])) for item in all_results
        ),
        'processing_stats': controller.get_processing_stats()
    }

    final_result = {
        'results': all_results,
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    print(f"\n处理完成: {successful_count}/{len(video_files)} 成功")

    return final_result


def save_final_results(controller: MainController, results: Dict[str, Any], is_single_file: bool = False) -> None:
    """
    保存最终结果

    Args:
        controller: 主控制器
        results: 处理结果
        is_single_file: 是否为单个文件处理
    """
    try:
        if is_single_file:
            # 单个文件的结果已经在处理时保存
            print(f"\n结果已保存到输出目录: {controller.config.OUTPUT_DIR}")
        else:
            # 保存批量处理的结果汇总
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                controller.config.OUTPUT_DIR,
                f'reid_batch_results_{timestamp}.json'
            )

            if controller.save_result(results, output_path):
                print(f"\n批量处理结果已保存: {output_path}")

    except Exception as e:
        print(f"保存结果失败: {e}")


def main() -> int:
    """
    主函数

    Returns:
        退出代码 (0=成功, 非0=失败)
    """
    try:
        # 打印横幅
        print_banner()

        # 解析命令行参数
        args = parse_arguments()

        # 验证参数
        if not validate_arguments(args):
            return 1

        # 设置日志
        setup_logging(args.log_level, args.log_file)

        # 创建配置对象
        config = Config()

        # 根据命令行参数更新配置
        update_config_from_args(config, args)

        # 验证配置
        config.validate_config()

        # 打印配置信息
        print_configuration(config, args)

        # 创建主控制器
        print("\n初始化系统...")
        controller = MainController(config)

        if not controller.initialize():
            print("✗ 系统初始化失败")
            return 1

        print("✓ 系统初始化成功")

        # 开始处理
        start_time = time.time()

        if args.single:
            # 处理单个视频文件
            results = process_single_video_file(controller, args.single)
            save_final_results(controller, results, is_single_file=True)
        else:
            # 处理视频目录
            input_dir = args.input_dir or config.VIDEO_INPUT_DIR
            results = process_video_directory(controller, input_dir, args.extensions)
            save_final_results(controller, results, is_single_file=False)

        # 打印最终摘要
        total_time = time.time() - start_time
        print(f"\n总处理时间: {total_time:.2f}秒")

        controller.print_summary()

        return 0

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 130

    except Exception as e:
        print(f"\n程序执行失败: {e}")
        logger.error(f"程序执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # 设置控制台编码（Windows系统）
    if os.name == 'nt':
        os.system('chcp 65001 > nul 2>&1')  # 设置UTF-8编码

    # 运行主程序
    exit_code = main()

    # 退出程序
    sys.exit(exit_code)