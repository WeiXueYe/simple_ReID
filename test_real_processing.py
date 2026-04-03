#!/usr/bin/env python3
"""
Test real video processing to debug the 0.000 similarity issue
"""

import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_real_video_processing():
    """Test processing a few frames from the actual video"""
    print("=== Real Video Processing Test ===\n")

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

        # Test video processing with just a few frames
        video_path = "./video/1.mp4"

        # Extract just first few frames for testing
        print(f"\nTesting with video: {video_path}")

        # Open the video
        if not controller.video_processor.open_video(video_path):
            print("FAIL Could not open video")
            return False

        frame_count = 0
        max_frames = 50  # Only process first 50 frames for debugging

        for frame, frame_number, timestamp_ms in controller.video_processor.extract_frames():
            if frame_count >= max_frames:
                break

            print(f"\n--- Frame {frame_number} ---")

            # Test detection
            detections = controller.person_detector.detect_persons(frame)
            print(f"Detections: {len(detections)}")

            if detections:
                # Test feature extraction for first detection
                detection = detections[0]
                bbox = detection['bbox']

                # Crop person image
                person_image = controller._crop_person_image(frame, bbox)
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
            else:
                print("No detections in this frame")

            frame_count += 1

            # Process just a few frames for debugging
            if frame_count >= 5:
                break

        print(f"\nProcessed {frame_count} frames")
        return True

    except Exception as e:
        print(f"FAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing Real Video Processing\n")

    success = test_real_video_processing()

    print(f"\n=== Test Summary ===")
    print(f"Real video processing: {'PASS' if success else 'FAIL'}")

    if success:
        print("\nSUCCESS All tests passed!")
        return 0
    else:
        print("\nFAIL Some tests failed - see output above")
        return 1

if __name__ == "__main__":
    exit(main())