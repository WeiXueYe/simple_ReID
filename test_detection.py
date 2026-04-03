#!/usr/bin/env python3
"""
Test detection with different confidence thresholds
"""

import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_detection_thresholds():
    """Test detection with different confidence thresholds"""
    print("=== Detection Threshold Test ===\n")

    try:
        from src.main_controller import MainController
        from src.config import Config

        # Initialize controller
        config = Config()
        controller = MainController(config)

        if not controller.initialize():
            print("FAIL Controller initialization failed")
            return False

        print("PASS Controller initialized successfully")

        # Open the video
        video_path = "./video/1.mp4"
        if not controller.video_processor.open_video(video_path):
            print("FAIL Could not open video")
            return False

        # Get first frame with people
        frame_count = 0
        test_frame = None

        for frame, frame_number, timestamp_ms in controller.video_processor.extract_frames():
            # Test detection with low threshold to find a frame with people
            detections = controller.person_detector.detect_persons(frame, confidence_threshold=0.1)
            if detections:
                test_frame = frame
                print(f"Found frame {frame_number} with {len(detections)} detections at low threshold")
                break

            frame_count += 1
            if frame_count > 100:  # Don't search too long
                break

        if test_frame is None:
            print("FAIL Could not find any frame with people")
            return False

        # Test different confidence thresholds
        thresholds = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]

        print(f"\nTesting different confidence thresholds on frame {frame_count}:")
        for threshold in thresholds:
            detections = controller.person_detector.detect_persons(test_frame, confidence_threshold=threshold)
            print(f"  Threshold {threshold}: {len(detections)} detections")

            if detections and threshold == 0.1:  # Show details for low threshold
                for i, det in enumerate(detections[:3]):  # Show first 3
                    print(f"    Detection {i}: conf={det['confidence']:.3f}, bbox={det['bbox']}")

        # Test feature extraction on detections
        print(f"\nTesting feature extraction on frame {frame_count}:")
        detections = controller.person_detector.detect_persons(test_frame, confidence_threshold=0.1)

        if detections:
            # Crop first person
            detection = detections[0]
            bbox = detection['bbox']
            person_image = controller._crop_person_image(test_frame, bbox)

            if person_image is not None:
                print(f"Person image shape: {person_image.shape}")

                # Extract feature
                feature = controller.feature_extractor.extract_single_feature(person_image)

                if feature is not None:
                    print(f"Feature shape: {feature.shape}")
                    print(f"Feature norm: {np.linalg.norm(feature):.6f}")
                    print(f"Feature min/max: {feature.min():.6f}/{feature.max():.6f}")
                    print(f"Feature mean: {feature.mean():.6f}")
                    print(f"Any NaN: {np.isnan(feature).any()}")
                    print(f"Any Inf: {np.isinf(feature).any()}")

                    # Test similarity with itself
                    from src.utils import cosine_similarity
                    self_sim = cosine_similarity(feature, feature)
                    print(f"Self similarity: {self_sim:.6f}")
                else:
                    print("FAIL Feature extraction returned None")
            else:
                print("FAIL Could not crop person image")

        return True

    except Exception as e:
        print(f"FAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing Detection Thresholds\n")

    success = test_detection_thresholds()

    print(f"\n=== Test Summary ===")
    print(f"Detection threshold test: {'PASS' if success else 'FAIL'}")

    if success:
        print("\nSUCCESS All tests passed!")
        return 0
    else:
        print("\nFAIL Some tests failed - see output above")
        return 1

if __name__ == "__main__":
    exit(main())