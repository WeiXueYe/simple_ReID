#!/usr/bin/env python3
"""
Debug similarity calculation in the actual tracking process
"""

import os
import sys
import numpy as np

# Add paths and set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_similarity_calculation():
    """Debug similarity calculation in tracking"""
    print("=== Debug Similarity Calculation ===\n")

    try:
        from src.main_controller import MainController
        from src.config import Config
        from src.utils import cosine_similarity

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

        # Process first few frames and collect features
        features = []
        frame_features = []

        frame_count = 0
        for frame, frame_number, timestamp_ms in controller.video_processor.extract_frames():
            if frame_count > 10:  # Process first 10 frames
                break

            print(f"\n--- Frame {frame_number} ---")

            # Get detections with low threshold
            detections = controller.person_detector.detect_persons(frame, confidence_threshold=0.1)
            print(f"Detections: {len(detections)}")

            if detections:
                # Extract features for all detections
                person_images = []
                for detection in detections:
                    bbox = detection['bbox']
                    person_image = controller._crop_person_image(frame, bbox)
                    if person_image is not None:
                        person_images.append(person_image)

                if person_images:
                    frame_feats = controller.feature_extractor.extract_features(person_images)
                    print(f"Extracted {len(frame_feats)} features")

                    # Store features for comparison
                    for i, feat in enumerate(frame_feats):
                        if feat is not None:
                            print(f"  Feature {i}: shape={feat.shape}, norm={np.linalg.norm(feat):.6f}")
                            print(f"    min={feat.min():.6f}, max={feat.max():.6f}, mean={feat.mean():.6f}")

                            # Test self-similarity
                            self_sim = cosine_similarity(feat, feat)
                            print(f"    Self-similarity: {self_sim:.6f}")

                            # Compare with previous features
                            if features:
                                for j, prev_feat in enumerate(features):
                                    if prev_feat is not None:
                                        sim = cosine_similarity(feat, prev_feat)
                                        print(f"    Similarity with feature {j}: {sim:.6f}")

                            features.append(feat)

            frame_count += 1

        print(f"\nCollected {len(features)} features total")

        # Test all pairwise similarities
        print(f"\nAll pairwise similarities:")
        valid_features = [f for f in features if f is not None]
        for i in range(len(valid_features)):
            for j in range(i+1, len(valid_features)):
                sim = cosine_similarity(valid_features[i], valid_features[j])
                print(f"Feature {i} vs {j}: {sim:.6f}")

        return True

    except Exception as e:
        print(f"FAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("Debugging Similarity Calculation\n")

    success = debug_similarity_calculation()

    print(f"\n=== Debug Summary ===")
    print(f"Similarity calculation: {'PASS' if success else 'FAIL'}")

    if success:
        print("\nSUCCESS Debug completed!")
        return 0
    else:
        print("\nFAIL Debug failed")
        return 1

if __name__ == "__main__":
    exit(main())