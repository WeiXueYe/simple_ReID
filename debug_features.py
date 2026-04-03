#!/usr/bin/env python3
"""
Debug script to test feature extraction and similarity calculation
"""

import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set encoding for Windows
if os.name == 'nt':
    os.system('chcp 65001 > nul 2>&1')  # UTF-8 encoding

def test_feature_extraction():
    """Test feature extraction and similarity calculation"""
    print("=== Feature Extraction Debug ===\n")

    try:
        # Import modules - use absolute imports
        from src.feature_extractor import FeatureExtractor
        from src.person_tracker import PersonTracker
        from src.config import Config
        from src.utils import cosine_similarity

        # Initialize components
        config = Config()
        print("Config loaded successfully")

        # Test feature extractor
        extractor = FeatureExtractor(config)
        if not extractor.initialize():
            print("FAIL Feature extractor initialization failed")
            return False
        print("PASS Feature extractor initialized")

        # Create test images (simple colored squares)
        print("\nCreating test images...")

        # Test 1: Simple colored images
        red_image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        blue_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        red_image2 = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)

        print("Extracting features from test images...")

        try:
            # Extract features
            features = extractor.extract_features([red_image, blue_image, red_image2])

            if not features:
                print("FAIL No features extracted")
                return False

            print(f"PASS Extracted {len(features)} features")

            # Check feature properties
            for i, feature in enumerate(features):
                if feature is None:
                    print(f"FAIL Feature {i} is None")
                    continue

                print(f"Feature {i}:")
                print(f"  Shape: {feature.shape}")
                print(f"  Dtype: {feature.dtype}")
                print(f"  Min: {feature.min():.6f}")
                print(f"  Max: {feature.max():.6f}")
                print(f"  Mean: {feature.mean():.6f}")
                print(f"  Norm: {np.linalg.norm(feature):.6f}")
                print(f"  Any NaN: {np.isnan(feature).any()}")
                print(f"  Any Inf: {np.isinf(feature).any()}")

            # Test similarity calculations
            print("\nTesting similarity calculations...")

            # Similar images should have high similarity
            sim_same = cosine_similarity(features[0], features[2])  # red vs red
            sim_diff = cosine_similarity(features[0], features[1])  # red vs blue

            print(f"Red vs Red similarity: {sim_same:.6f}")
            print(f"Red vs Blue similarity: {sim_diff:.6f}")

            # Test edge cases
            print("\nTesting edge cases...")

            # Zero vector
            zero_vec = np.zeros_like(features[0])
            sim_zero = cosine_similarity(features[0], zero_vec)
            print(f"Feature vs zero vector: {sim_zero:.6f}")

            # Identical vectors
            sim_identical = cosine_similarity(features[0], features[0])
            print(f"Feature vs itself: {sim_identical:.6f}")

            # Check if similarities are reasonable
            if sim_same > 0.5:  # Same color images should be somewhat similar
                print("PASS Same color similarity looks reasonable")
            else:
                print("FAIL Same color similarity too low")

            if sim_identical > 0.99:  # Identical should be very close to 1
                print("PASS Self similarity looks good")
            else:
                print("FAIL Self similarity too low")

            return True

        except Exception as e:
            print(f"FAIL Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"FAIL Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_person_tracker():
    """Test person tracker with debug info"""
    print("\n=== Person Tracker Debug ===\n")

    try:
        from src.person_tracker import PersonTracker
        from src.config import Config

        config = Config()
        tracker = PersonTracker(config)

        if not tracker.initialize():
            print("FAIL Person tracker initialization failed")
            return False

        print("PASS Person tracker initialized")

        # Test similarity threshold logic
        normal_threshold = tracker._get_current_similarity_threshold()
        print(f"Normal threshold: {normal_threshold}")

        tracker.is_post_transition = True
        transition_threshold = tracker._get_current_similarity_threshold()
        print(f"Transition threshold: {transition_threshold}")

        tracker.is_post_transition = False

        # Test feature weight logic
        normal_weight = tracker._get_current_feature_weight()
        print(f"Normal feature weight: {normal_weight}")

        tracker.is_post_transition = True
        transition_weight = tracker._get_current_feature_weight()
        print(f"Transition feature weight: {transition_weight}")

        return True

    except Exception as e:
        print(f"FAIL Person tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("Debugging ReID Feature Extraction and Similarity\n")

    success1 = test_feature_extraction()
    success2 = test_person_tracker()

    print(f"\n=== Debug Summary ===")
    print(f"Feature extraction: {'PASS PASS' if success1 else 'FAIL FAIL'}")
    print(f"Person tracker: {'PASS PASS' if success2 else 'FAIL FAIL'}")

    if success1 and success2:
        print("\nSUCCESS All tests passed!")
        return 0
    else:
        print("\nFAIL Some tests failed - see output above")
        return 1

if __name__ == "__main__":
    exit(main())