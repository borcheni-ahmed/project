from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque
from threading import Thread, Lock
import serial

# ----------------------------
# Motor control (with serial to Arduino)
# ----------------------------
class MotorController:
    def __init__(self, port="/dev/ttyACM0", baud=9600):
        self.state = "STOP"
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            print("[SERIAL] Connected to Arduino.")
        except Exception as e:
            print(f"[ERROR] Serial connection failed: {e}")
            self.ser = None

    def _send(self, cmd):
        print(f"[MOTORS] {cmd}")
        if self.ser:
            try:
                self.ser.write((cmd + "\n").encode())
            except Exception as e:
                print(f"[SERIAL ERROR] {e}")

    def stop(self):
        if self.state != "STOP":
            self._send("STOP")
            self.state = "STOP"

    def forward(self):
        if self.state != "FORWARD":
            self._send("FORWARD")
            self.state = "FORWARD"

    def left(self):
        if self.state != "LEFT":
            self._send("LEFT")
            self.state = "LEFT"

    def right(self):
        if self.state != "RIGHT":
            self._send("RIGHT")
            self.state = "RIGHT"

    def backward(self):
        if self.state != "BACKWARD":
            self._send("BACKWARD")
            self.state = "BACKWARD"

# ----------------------------
# Configuration
# ----------------------------
class Config:
    FRAME_WIDTH = 480
    FRAME_HEIGHT = 360
    FPS = 30

    MIN_OBSTACLE_AREA = 150
    MAX_OBSTACLE_AREA = 100000

    CRITICAL_PROXIMITY = 45
    WARNING_PROXIMITY = 30
    SAFE_PROXIMITY = 15

    ROI_TOP_RATIO = 0.05
    ROI_BOTTOM_RATIO = 0.95
    ROI_LEFT_RATIO = 0.05
    ROI_RIGHT_RATIO = 0.95

    OBSTACLE_HISTORY_SIZE = 3
    MIN_DETECTIONS_REQUIRED = 2

# ----------------------------
# Threaded camera
# ----------------------------
class ThreadedCamera:
    def __init__(self, width=480, height=360):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
            controls={"AwbEnable": True, "AeEnable": True, "FrameRate": Config.FPS}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        self.frame = None
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                frame = self.picam2.capture_array()
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.frame = bgr
            except:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()

# ----------------------------
# Proximity estimator
# ----------------------------
class ProximityEstimator:
    @staticmethod
    def calculate_proximity(bbox, area, frame_height, frame_width):
        x, y, w, h = bbox
        area_score = min(100, (area / Config.MAX_OBSTACLE_AREA) * 150)
        normalized_y = (y + h / 2) / frame_height
        position_score = (1 - normalized_y) * 100
        width_score = min(100, (w / frame_width) * 100)
        proximity = (area_score * 0.5) + (position_score * 0.3) + (width_score * 0.2)
        return int(max(0, min(100, proximity)))

    @staticmethod
    def proximity_to_level(proximity):
        if proximity >= Config.CRITICAL_PROXIMITY:
            return "CRITICAL", (0, 0, 255)
        elif proximity >= Config.WARNING_PROXIMITY:
            return "CLOSE", (0, 165, 255)
        elif proximity >= Config.SAFE_PROXIMITY:
            return "MEDIUM", (0, 255, 255)
        else:
            return "SAFE", (0, 255, 0)

# ----------------------------
# Enhanced all-object detector
# ----------------------------
class EnhancedObstacleDetector:
    def __init__(self):
        self.prev_gray = None
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )

    def _detect_edges(self, gray):
        """Multi-scale edge detection for better obstacle boundaries"""
        # Canny edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 30, 90)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.dilate(edges, self.kernel_small, iterations=1)
        return edges

    def _detect_texture(self, gray):
        """Detect texture changes that indicate objects"""
        # Laplacian for texture detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, texture_mask = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        return texture_mask

    def _detect_depth_cues(self, gray):
        """Use brightness and contrast for depth perception"""
        # Adaptive thresholding to find objects with different brightness
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Also detect very dark and very bright regions
        _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        depth_mask = cv2.bitwise_or(adaptive, cv2.bitwise_or(dark, bright))
        return depth_mask

    def _detect_motion(self, frame, gray):
        """Detect moving objects using background subtraction and frame differencing"""
        # Background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Frame differencing for fast motion
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_combined = cv2.bitwise_or(fg_mask, motion_mask)
        else:
            motion_combined = fg_mask
        
        self.prev_gray = gray.copy()
        return motion_combined

    def _detect_contours_advanced(self, gray):
        """Detect object contours using morphological operations"""
        # Morphological gradient to find object boundaries
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, self.kernel_small)
        _, gradient_thresh = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        
        # Close gaps in boundaries
        closed = cv2.morphologyEx(gradient_thresh, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        return closed

    def detect(self, frame):
        h, w = frame.shape[:2]
        roi_top = int(h * Config.ROI_TOP_RATIO)
        roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
        roi_left = int(w * Config.ROI_LEFT_RATIO)
        roi_right = int(w * Config.ROI_RIGHT_RATIO)
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # MULTIPLE DETECTION METHODS
        # 1. Edge detection (static objects with clear boundaries)
        edges = self._detect_edges(enhanced)
        
        # 2. Texture detection (objects with different texture)
        texture = self._detect_texture(enhanced)
        
        # 3. Depth cues (objects with different brightness)
        depth = self._detect_depth_cues(enhanced)
        
        # 4. Motion detection (moving objects)
        motion = self._detect_motion(roi, enhanced)
        
        # 5. Contour-based detection (solid objects)
        contours_mask = self._detect_contours_advanced(enhanced)

        # COMBINE ALL METHODS
        # Static detection (edges + texture + depth + contours)
        static_combined = cv2.bitwise_or(edges, texture)
        static_combined = cv2.bitwise_or(static_combined, depth)
        static_combined = cv2.bitwise_or(static_combined, contours_mask)
        
        # Combine with motion
        final_mask = cv2.bitwise_or(static_combined, motion)
        
        # Clean up noise
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        
        # Fill holes in detected objects
        final_mask = cv2.dilate(final_mask, self.kernel_small, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < Config.MIN_OBSTACLE_AREA or area > Config.MAX_OBSTACLE_AREA:
                continue
            
            # Get bounding box
            x, y, w_rect, h_rect = cv2.boundingRect(c)
            
            # Filter out noise (very thin or very wide objects)
            aspect_ratio = w_rect / float(h_rect) if h_rect > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            # Map back to original frame coordinates
            x_orig, y_orig = x + roi_left, y + roi_top
            bbox = (x_orig, y_orig, w_rect, h_rect)
            
            # Calculate proximity
            proximity = ProximityEstimator.calculate_proximity(bbox, area, h, w)
            cx, cy = x_orig + w_rect // 2, y_orig + h_rect // 2
            
            obstacles.append({
                'bbox': bbox, 
                'center': (cx, cy), 
                'proximity': proximity,
                'area': area
            })

        # Sort by proximity (closest first)
        obstacles.sort(key=lambda x: x['proximity'], reverse=True)
        return obstacles, final_mask

# ----------------------------
# Obstacle tracker
# ----------------------------
class StrictObstacleTracker:
    def __init__(self, history_size=3):
        self.history = deque(maxlen=history_size)

    def update(self, obstacles):
        self.history.append(obstacles)

    def get_stable_obstacles(self):
        if len(self.history) < Config.MIN_DETECTIONS_REQUIRED:
            return []
        
        position_map = {}
        for obs_list in self.history:
            for obs in obs_list:
                # Grid-based grouping for stable detection
                gx, gy = round(obs['center'][0] / 30) * 30, round(obs['center'][1] / 30) * 30
                position_map.setdefault((gx, gy), []).append(obs)
        
        # Keep obstacles seen in multiple frames
        stable = [v[-1] for v in position_map.values() if len(v) >= Config.MIN_DETECTIONS_REQUIRED]
        return stable

# ----------------------------
# Smart motor decision logic
# ----------------------------
class SmartMotorController:
    def __init__(self):
        self.motors = MotorController()

    def decide(self, obstacles):
        if not obstacles:
            self.motors.forward()
            return
        
        closest = obstacles[0]
        x_center = closest['center'][0]
        proximity = closest['proximity']

        if proximity >= Config.CRITICAL_PROXIMITY:
            if x_center < Config.FRAME_WIDTH / 3:
                self.motors.right()
            elif x_center > Config.FRAME_WIDTH * 2 / 3:
                self.motors.left()
            else:
                self.motors.stop()
        elif proximity >= Config.WARNING_PROXIMITY:
            self.motors.stop()
        else:
            self.motors.forward()

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("[SYSTEM] Starting Enhanced ADAS with All-Object Detection...")
    camera = ThreadedCamera(Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    detector = EnhancedObstacleDetector()
    tracker = StrictObstacleTracker(Config.OBSTACLE_HISTORY_SIZE)
    controller = SmartMotorController()

    show_debug = False
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            obstacles, mask = detector.detect(frame)
            tracker.update(obstacles)
            stable = tracker.get_stable_obstacles()
            controller.decide(stable)

            display = frame.copy()
            for obs in stable:
                x, y, w_rect, h_rect = obs['bbox']
                level, color = ProximityEstimator.proximity_to_level(obs['proximity'])
                cv2.rectangle(display, (x, y), (x + w_rect, y + h_rect), color, 2)
                cv2.putText(display, f"{obs['proximity']}% {level}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            cv2.putText(display, f"FPS:{current_fps} Det:{len(obstacles)} Stable:{len(stable)}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("ADAS - Enhanced Detection", display)

            if show_debug:
                debug_resized = cv2.resize(mask, (320, 240))
                cv2.imshow("Debug Mask", debug_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Debug Mask")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        controller.motors.stop()
        print("[SYSTEM] Shutdown complete")

if __name__ == "__main__":
    main()
