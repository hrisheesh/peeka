"""Training capture module for facial feature detection and recording.

This module handles the capture and processing of facial features during the training phase,
including facial landmark detection and data recording.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import wave
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION

from .video import VideoCapture
from .audio import AudioCapture
import time
import random

class TrainingDataCapture:
    """Handles synchronized capture of video and audio training data."""

    def __init__(self):
        """Initialize training capture with required components."""
        try:
            self.video_capture = VideoCapture()
            self.audio_capture = AudioCapture()
            self.frame_count = 0
            self.target_duration = 300  # 5 minutes in seconds
            self.recording = False
            self.output_dir = None
            self.face_mesh = None
            self.last_detection_time = 0  # Track last detection time
            self.detection_interval = 0.1  # 100ms minimum interval between detections
            self.audio_frames = []
            self._initialize_detectors()
        except Exception as e:
            print(f"Failed to initialize capture components: {str(e)}")
            raise RuntimeError(f"Failed to initialize capture components: {str(e)}")

    def _initialize_detectors(self) -> None:
        """Initialize MediaPipe Face Mesh for facial landmark detection."""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,  # Set to False for better tracking
                max_num_faces=1,
                min_detection_confidence=0.5,  # Increased threshold for more reliable detection
                min_tracking_confidence=0.5,  # Increased threshold for more stable tracking
                refine_landmarks=False  # Match TrainingWindow configuration for consistency
            )
            # Test the face mesh initialization with a proper test frame
            test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray test frame
            test_result = self.face_mesh.process(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))  # Convert to RGB for testing
            if test_result.multi_face_landmarks is None:
                print("Warning: Face mesh initialization test failed, but continuing")
        except Exception as e:
            print(f"Failed to initialize facial detection: {str(e)}")
            self.face_mesh = None
            raise RuntimeError(f"Failed to initialize facial detection: {str(e)}. Please ensure mediapipe is properly installed.")

    def detect_facial_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect facial landmarks with improved reliability."""
        if self.face_mesh is None:
            return None

        try:
            if frame is None or frame.size == 0:
                return None

            # Get image dimensions for MediaPipe
            img_h, img_w = frame.shape[:2]
            
            # Convert to RGB and process with dimensions
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None

            # Get primary face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Create normalized landmarks first
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # Convert to pixel coordinates after
            landmarks_2d = np.array([
                [landmark[0] * img_w, landmark[1] * img_h] 
                for landmark in landmarks_norm[:, :2]
            ])
            
            landmarks_3d = np.array([
                [landmark[0] * img_w, landmark[1] * img_h, landmark[2] * img_w] 
                for landmark in landmarks_norm
            ])

            # Calculate face metrics using normalized coordinates
            face_metrics = self._calculate_face_metrics(landmarks_2d, landmarks_3d)
            
            # Store both normalized and pixel coordinates
            return {
                'landmarks_2d': landmarks_2d.tolist(),  # Convert to list for JSON serialization
                'landmarks_3d': landmarks_3d.tolist(),
                'landmarks_normalized': landmarks_norm.tolist(),
                'face_metrics': face_metrics,
                'image_size': {'width': img_w, 'height': img_h}
            }

        except Exception as e:
            print(f"Landmark detection error: {str(e)}")
            return None

    def _calculate_face_metrics(self, landmarks_2d: np.ndarray, landmarks_3d: np.ndarray) -> Dict:
        """Calculate detailed facial metrics from landmarks.

        Args:
            landmarks_2d: 2D facial landmarks array
            landmarks_3d: 3D facial landmarks array

        Returns:
            Dict: Facial metrics including distances, ratios and depth information
        """
        metrics = {}

        # MediaPipe face mesh indices for key facial features
        left_eye = landmarks_2d[[33, 133, 157, 158, 159, 160, 161, 246]]
        right_eye = landmarks_2d[[362, 263, 386, 387, 388, 389, 390, 466]]
        mouth_outer = landmarks_2d[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]]
        mouth_inner = landmarks_2d[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]]
        left_eyebrow = landmarks_2d[[276, 283, 282, 295, 285]]
        right_eyebrow = landmarks_2d[[46, 53, 52, 65, 55]]

        # Eye aspect ratios (including depth)
        def eye_aspect_ratio_3d(eye_points_2d, eye_points_3d):
            width = np.linalg.norm(eye_points_2d[0] - eye_points_2d[4])
            height = (np.linalg.norm(eye_points_2d[2] - eye_points_2d[6]) + 
                     np.linalg.norm(eye_points_2d[1] - eye_points_2d[5])) / 2
            depth = np.mean([point[2] for point in eye_points_3d])
            return height / width, depth

        metrics['left_eye_ratio'], metrics['left_eye_depth'] = eye_aspect_ratio_3d(
            left_eye, landmarks_3d[[33, 133, 157, 158, 159, 160, 161, 246]])
        metrics['right_eye_ratio'], metrics['right_eye_depth'] = eye_aspect_ratio_3d(
            right_eye, landmarks_3d[[362, 263, 386, 387, 388, 389, 390, 466]])

        # Mouth metrics with depth
        mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])
        mouth_height = np.linalg.norm(mouth_outer[3] - mouth_outer[9])
        mouth_depth = np.mean([landmarks_3d[i][2] for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]])
        metrics['mouth_aspect_ratio'] = mouth_height / mouth_width
        metrics['mouth_depth'] = mouth_depth

        # Eyebrow positions and depth
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        metrics['left_brow_height'] = np.mean(left_eyebrow[:, 1]) - left_eye_center[1]
        metrics['right_brow_height'] = np.mean(right_eyebrow[:, 1]) - right_eye_center[1]
        metrics['left_brow_depth'] = np.mean([landmarks_3d[i][2] for i in [276, 283, 282, 295, 285]])
        metrics['right_brow_depth'] = np.mean([landmarks_3d[i][2] for i in [46, 53, 52, 65, 55]])

        return metrics

    def _assess_frame_quality(self, frame):
        """Assess overall frame quality."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check brightness
        mean_brightness = np.mean(gray)
        brightness_ok = 20 <= mean_brightness <= 235
        
        # Check contrast
        contrast = np.std(gray)
        contrast_ok = contrast > 20
        
        # Check blur
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_ok = laplacian > 100
        
        return {
            'brightness': mean_brightness,
            'contrast': contrast,
            'sharpness': laplacian,
            'quality_score': sum([brightness_ok, contrast_ok, blur_ok]) / 3.0
        }

    def start_recording(self) -> Tuple[bool, str]:
        """Start the training capture session.

        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            # Release any existing capture
            if hasattr(self.video_capture, 'cap') and self.video_capture.cap is not None:
                self.video_capture.release()

            # Initialize video capture first
            if not self.video_capture.start():
                return False, "Failed to start video capture"

            # Try to initialize audio capture, but continue if it fails
            try:
                if not self.audio_capture.start():
                    print("Warning: Failed to start audio capture, continuing with video only")
            except Exception as audio_error:
                print(f"Warning: Audio capture error: {str(audio_error)}")

            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                          'training_data', f'session_{timestamp}')
            os.makedirs(os.path.join(self.output_dir, 'face_frames'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'audio_data'), exist_ok=True)

            self.frame_count = 0
            self.audio_frames = []
            self.recording = True
            return True, "Recording started successfully"

        except Exception as e:
            return False, str(e)

    def stop_recording(self) -> str:
        """Stop the training capture session and save audio data.

        Returns:
            str: Path to the session directory
        """
        self.recording = False
        self.video_capture.release()
        self.audio_capture.stop()

        # Save audio data
        if self.audio_frames:
            audio_file = os.path.join(self.output_dir, 'audio_data', 'audio.wav')
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.audio_frames))

        return self.output_dir

    def update_capture(self, frame: np.ndarray) -> None:
        """Process and save the current frame and audio data.

        Args:
            frame: Current video frame
        """
        if not self.recording or frame is None:
            return

        try:
            # Save raw frame first
            frame_path = os.path.join(self.output_dir, 'face_frames', f'frame_{self.frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            self.frame_count += 1

            # Capture audio frame
            audio_frame = self.audio_capture.read_frame()
            if audio_frame is not None:
                self.audio_frames.append(audio_frame)

            # Detect and save facial landmarks if possible
            landmarks = self.detect_facial_landmarks(frame)
            if landmarks is not None:
                # Save landmarks data with audio timestamp
                landmarks_data = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_id': self.frame_count - 1,
                    'audio_frame_index': len(self.audio_frames) - 1,
                    'landmarks_2d': landmarks['landmarks_2d'],
                    'landmarks_3d': landmarks['landmarks_3d'],
                    'landmarks_normalized': landmarks['landmarks_normalized'],
                    'face_metrics': landmarks['face_metrics'],
                    'image_size': landmarks['image_size'],
                    'frame_quality': self._assess_frame_quality(frame)
                }
                
                # Save landmarks data
                landmarks_dir = os.path.join(self.output_dir, 'landmarks_data')
                os.makedirs(landmarks_dir, exist_ok=True)
                landmarks_path = os.path.join(landmarks_dir, f'frame_{self.frame_count-1:06d}.json')
                
                import json
                with open(landmarks_path, 'w') as f:
                    json.dump(landmarks_data, f, indent=2)

        except Exception as e:
            print(f"Error processing frame: {e}")

    def validate_recording(self, session_dir: str) -> bool:
        """Validate recorded training data with improved metrics."""
        try:
            face_frames_dir = os.path.join(session_dir, 'face_frames')
            landmarks_dir = os.path.join(session_dir, 'landmarks_data')
            
            if not os.path.exists(face_frames_dir) or not os.path.exists(landmarks_dir):
                print("Missing required directories")
                return False

            frame_files = sorted([f for f in os.listdir(face_frames_dir) if f.endswith('.jpg')])
            
            if len(frame_files) < 60:  # Minimum 2 seconds at 30 FPS
                print(f"Insufficient frames: {len(frame_files)}")
                return False

            # Process frames in sequences
            sequence_length = 30  # 1 second at 30 FPS
            valid_sequences = 0
            total_sequences = 0
            
            for i in range(0, len(frame_files), sequence_length):
                sequence = frame_files[i:i + sequence_length]
                sequence_metrics = self._validate_sequence(
                    [os.path.join(face_frames_dir, f) for f in sequence]
                )
                
                if sequence_metrics['valid'] and sequence_metrics['landmarks_present']:
                    valid_sequences += 1
                total_sequences += 1

                # Print detailed metrics for debugging
                print(f"Sequence {total_sequences} metrics:")
                print(f"- Valid frames: {sequence_metrics['valid_frame_ratio']*100:.1f}%")
                print(f"- Quality score: {sequence_metrics['average_quality']:.2f}")
                print(f"- Landmarks present: {sequence_metrics['landmarks_present']}")

            validation_ratio = valid_sequences / total_sequences
            print(f"Final validation ratio: {validation_ratio:.2f}")
            
            return validation_ratio >= 0.5  # At least 50% of sequences must be valid

        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def _validate_sequence(self, frame_paths: List[str]) -> Dict:
        """Validate a sequence of frames with improved error handling."""
        valid_frames = 0
        quality_scores = []
        landmarks_data = []

        for frame_path in frame_paths:
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                result = self.detect_facial_landmarks(frame)
                if result is None:
                    continue

                # Ensure we have valid landmark data
                if not all(key in result for key in ['landmarks_2d', 'landmarks_normalized', 'face_metrics']):
                    print(f"Missing required landmark data in frame {frame_path}")
                    continue

                # Convert lists back to numpy arrays for processing
                landmarks_2d = np.array(result['landmarks_2d'])
                landmarks_norm = np.array(result['landmarks_normalized'])

                # Calculate quality score
                quality_score = self._calculate_quality_score(frame, landmarks_2d)
                quality_scores.append(quality_score)

                # Store landmark data for stability check
                landmarks_data.append({
                    'landmarks_2d': landmarks_2d,
                    'landmarks_normalized': landmarks_norm,
                    'face_metrics': result['face_metrics']
                })
                valid_frames += 1

            except Exception as e:
                print(f"Error processing frame {frame_path}: {str(e)}")
                continue

        # Calculate sequence metrics
        sequence_valid = False
        if valid_frames >= len(frame_paths) * 0.5:  # Increased threshold to 50%
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            if landmarks_data:
                stability = self._check_landmark_stability(landmarks_data)
                sequence_valid = avg_quality >= 0.5 and stability >= 0.6  # Stricter thresholds

        return {
            'valid': sequence_valid,
            'valid_frame_ratio': valid_frames / len(frame_paths),
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'landmarks_present': len(landmarks_data) > 0
        }

    def _calculate_quality_score(self, frame: np.ndarray, landmarks_2d: np.ndarray) -> float:
        """Calculate frame quality score based on multiple factors."""
        try:
            # Check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_score = min(1.0, max(0.0, np.mean(gray) / 127.5))

            # Check face size relative to frame
            face_width = np.max(landmarks_2d[:, 0]) - np.min(landmarks_2d[:, 0])
            face_height = np.max(landmarks_2d[:, 1]) - np.min(landmarks_2d[:, 1])
            frame_h, frame_w = frame.shape[:2]
            size_score = min(1.0, (face_width * face_height) / (frame_w * frame_h * 0.15))

            # Combine scores
            return (brightness_score + size_score) / 2.0

        except Exception as e:
            print(f"Error calculating quality score: {str(e)}")
            return 0.0

    def _check_landmark_stability(self, landmarks_data: List[Dict]) -> float:
        """Check stability of landmarks across frames."""
        try:
            if len(landmarks_data) < 2:
                return 0.0

            stability_scores = []
            for i in range(1, len(landmarks_data)):
                prev_landmarks = landmarks_data[i-1]['landmarks_2d']
                curr_landmarks = landmarks_data[i]['landmarks_2d']
                
                # Calculate movement between consecutive frames
                movement = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
                
                # Convert movement to stability score (inverse relationship)
                stability = 1.0 / (1.0 + movement/100)  # Normalize movement
                stability_scores.append(stability)

            return np.mean(stability_scores)
        except Exception as e:
            print(f"Error checking landmark stability: {e}")
            return 0.0