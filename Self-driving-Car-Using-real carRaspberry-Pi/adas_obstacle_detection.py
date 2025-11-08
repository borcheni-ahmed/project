from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque
from threading import Thread, Lock

# ----------------------------
# Motor control functions
# ----------------------------
def stop():
    print("[MOTORS] STOP")

def forward():
    print("[MOTORS] FORWARD")

def left():
    print("[MOTORS] LEFT")

def right():
    print("[MOTORS] RIGHT")

def backward():
    print("[MOTORS] BACKWARD")

# ----------------------------
# Configuration
# ----------------------------
class Config:
    # Camera settings (reduced for performance)
    FRAME_WIDTH = 480
    FRAME_HEIGHT = 360
    FPS = 30
    
    # Distance estimation calibration (adjust based on your setup)
    FOCAL_LENGTH = 500  # Calibration constant
    KNOWN_OBJECT_WIDTH = 20  # Average obstacle width in cm
    
    # Obstacle detection
    MIN_OBSTACLE_AREA = 800
    MAX_OBSTACLE_AREA = 60000
    
    # Distance thresholds (in cm)
    CRITICAL_DISTANCE = 30  # Very close
    WARNING_DISTANCE = 60   # Need to slow
    SAFE_DISTANCE = 100     # Just monitoring
    
    # ROI settings
    ROI_TOP_RATIO = 0.30
    ROI_BOTTOM_RATIO = 0.95
    
    # Traffic light detection
    MIN_CIRCULARITY = 0.6
    MIN_LIGHT_AREA = 120
    MAX_LIGHT_AREA = 3000
    
    # Performance optimization
    SKIP_FRAMES = 0  # Process every frame now
    RESIZE_FOR_DETECTION = True  # Process smaller frame
    DETECTION_SCALE = 0.7  # Scale down for detection
    
    # History
    OBSTACLE_HISTORY_SIZE = 2
    LIGHT_HISTORY_SIZE = 2

# ----------------------------
# High-performance threaded camera
# ----------------------------
class ThreadedCamera:
    def __init__(self, width=480, height=360):
        self.picam2 = Picamera2()
        
        # Optimized low-latency configuration
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
            queue=False,  # Non-blocking
            controls={
                "AwbEnable": True,
                "AeEnable": True,
                "FrameRate": Config.FPS,
                "NoiseReductionMode": 0,  # Disable for speed
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        self.frame = None
        self.lock = Lock()
        self.running = True
        
        # Start high-priority capture thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        time.sleep(1.5)
    
    def _update(self):
        """Continuously capture frames"""
        while self.running:
            try:
                new_frame = self.picam2.capture_array()
                # Convert RGB to BGR immediately
                bgr_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.frame = bgr_frame
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
# Distance estimation
# ----------------------------
class DistanceEstimator:
    """Estimate distance to obstacles using perspective"""
    
    @staticmethod
    def estimate_from_bbox(bbox, frame_height):
        """Estimate distance based on bounding box size and position"""
        x, y, w, h = bbox
        
        # Method 1: Based on object height (more accurate)
        # Objects further away appear smaller
        if h > 0:
            # Simple inverse relationship: distance ∝ 1/height
            distance_height = (Config.FOCAL_LENGTH * Config.KNOWN_OBJECT_WIDTH) / h
        else:
            distance_height = 999
        
        # Method 2: Based on vertical position in frame
        # Objects lower in frame are closer
        y_center = y + h/2
        frame_bottom = frame_height * Config.ROI_BOTTOM_RATIO
        normalized_y = (frame_bottom - y_center) / frame_bottom
        distance_position = 150 * normalized_y  # Scale to reasonable range
        
        # Weighted average of both methods
        distance = (distance_height * 0.7) + (distance_position * 0.3)
        
        # Clamp to reasonable range
        distance = max(10, min(200, distance))
        
        return int(distance)

# ----------------------------
# Ultra-fast obstacle detector
# ----------------------------
class UltraFastObstacleDetector:
    def __init__(self):
        # Background subtractor with optimized settings
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=30, detectShadows=False
        )
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect(self, frame):
        """Ultra-fast multi-method obstacle detection with distance"""
        h, w = frame.shape[:2]
        
        # Define ROI
        roi_top = int(h * Config.ROI_TOP_RATIO)
        roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
        roi = frame[roi_top:roi_bottom, :]
        roi_h = roi.shape[0]
        
        # Optionally resize for faster processing
        if Config.RESIZE_FOR_DETECTION:
            process_roi = cv2.resize(roi, None, fx=Config.DETECTION_SCALE, 
                                     fy=Config.DETECTION_SCALE, 
                                     interpolation=cv2.INTER_LINEAR)
            scale_factor = 1.0 / Config.DETECTION_SCALE
        else:
            process_roi = roi
            scale_factor = 1.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(process_roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Fast edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Method 2: Background subtraction (motion)
        fg_mask = self.bg_subtractor.apply(process_roi, learningRate=0.003)
        _, motion = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)
        
        # Method 3: Adaptive thresholding for static objects
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 7, 2)
        
        # Combine all methods
        combined = cv2.bitwise_or(edges, motion)
        combined = cv2.bitwise_or(combined, thresh)
        
        # Quick cleanup
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel_small)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel_small)
        combined = cv2.dilate(combined, self.kernel_medium, iterations=1)
        
        # Find contours
        contours, hierarchy = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_OBSTACLE_AREA < area < Config.MAX_OBSTACLE_AREA:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Aspect ratio filter
                aspect = w_rect / float(h_rect) if h_rect > 0 else 0
                if 0.15 < aspect < 6.0:
                    # Scale back to original coordinates
                    x_orig = int(x * scale_factor)
                    y_orig = int(y * scale_factor) + roi_top
                    w_orig = int(w_rect * scale_factor)
                    h_orig = int(h_rect * scale_factor)
                    
                    # Calculate distance
                    bbox_roi = (x_orig - roi_top, y_orig - roi_top, w_orig, h_orig)
                    distance = DistanceEstimator.estimate_from_bbox(
                        (x_orig, y_orig - roi_top, w_orig, h_orig), 
                        h
                    )
                    
                    # Calculate center
                    cx = x_orig + w_orig // 2
                    cy = y_orig + h_orig // 2
                    
                    # Determine criticality based on distance
                    if distance < Config.CRITICAL_DISTANCE:
                        criticality = 100
                    elif distance < Config.WARNING_DISTANCE:
                        criticality = 70 - ((distance - Config.CRITICAL_DISTANCE) / 
                                           (Config.WARNING_DISTANCE - Config.CRITICAL_DISTANCE) * 40)
                    elif distance < Config.SAFE_DISTANCE:
                        criticality = 30 - ((distance - Config.WARNING_DISTANCE) / 
                                           (Config.SAFE_DISTANCE - Config.WARNING_DISTANCE) * 20)
                    else:
                        criticality = 10
                    
                    obstacles.append({
                        'bbox': (x_orig, y_orig, w_orig, h_orig),
                        'area': area * (scale_factor ** 2),
                        'center': (cx, cy),
                        'distance': distance,
                        'criticality': criticality
                    })
        
        # Sort by distance (closest first)
        obstacles.sort(key=lambda x: x['distance'])
        
        return obstacles[:8], combined  # Return top 8

# ----------------------------
# Optimized traffic light detector
# ----------------------------
class OptimizedTrafficLightDetector:
    def __init__(self):
        self.light_history = deque(maxlen=Config.LIGHT_HISTORY_SIZE)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect(self, frame):
        """Fast traffic light detection"""
        h, w = frame.shape[:2]
        
        # Small ROI for traffic lights (top-center)
        roi = frame[0:h//3, w//3:2*w//3]
        
        # Downsample for speed
        roi_small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
        
        # Quick color detection
        scores = {
            "RED": self._quick_detect(hsv, "RED"),
            "ORANGE": self._quick_detect(hsv, "ORANGE"),
            "GREEN": self._quick_detect(hsv, "GREEN")
        }
        
        # Get highest score
        max_score = max(scores.values())
        detected = None
        
        if max_score > 100:
            detected = max(scores, key=scores.get)
        
        # History smoothing
        self.light_history.append(detected)
        
        if len(self.light_history) >= 2:
            recent = list(self.light_history)[-2:]
            non_none = [x for x in recent if x is not None]
            if len(non_none) >= 1:
                return non_none[-1]
        
        return None
    
    def _quick_detect(self, hsv, color):
        """Fast color detection with circularity check"""
        if color == "RED":
            mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "ORANGE":
            mask = cv2.inRange(hsv, (8, 80, 80), (25, 255, 255))
        else:
            mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        
        # Quick cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Count pixels (faster than finding contours for scoring)
        score = cv2.countNonZero(mask)
        
        return score

# ----------------------------
# Lightweight tracker
# ----------------------------
class FastObstacleTracker:
    def __init__(self, history_size=2):
        self.history = deque(maxlen=history_size)
    
    def update(self, obstacles):
        self.history.append(obstacles)
    
    def get_stable_obstacles(self):
        if not self.history:
            return []
        return self.history[-1]  # Just return latest for speed

# ----------------------------
# Motor controller with distance-based logic
# ----------------------------
class SmartMotorController:
    def __init__(self):
        self.state = "STOP"
        self.last_avoidance_time = 0
        
    def decide(self, obstacles, traffic_light):
        current_time = time.time()
        
        # Priority 1: Red light
        if traffic_light == "RED":
            self._execute_stop()
            return
        
        # Priority 2: Obstacles by distance
        if obstacles:
            closest = obstacles[0]
            distance = closest['distance']
            
            if distance < Config.CRITICAL_DISTANCE:
                # Emergency avoidance
                if current_time - self.last_avoidance_time > 1.5:
                    self._execute_avoidance(closest)
                    self.last_avoidance_time = current_time
                return
            
            elif distance < Config.WARNING_DISTANCE:
                # Slow down / stop
                self._execute_stop()
                return
        
        # Priority 3: Orange light
        if traffic_light == "ORANGE":
            self._execute_stop()
            return
        
        # Clear path - go forward
        if traffic_light == "GREEN" or traffic_light is None:
            if not obstacles or obstacles[0]['distance'] > Config.SAFE_DISTANCE:
                self._execute_forward()
    
    def _execute_stop(self):
        if self.state != "STOP":
            stop()
            self.state = "STOP"
    
    def _execute_forward(self):
        if self.state != "FORWARD":
            forward()
            self.state = "FORWARD"
    
    def _execute_avoidance(self, obstacle):
        cx = obstacle['center'][0]
        distance = obstacle['distance']
        
        print(f"[AVOIDANCE] Distance: {distance}cm")
        
        stop()
        time.sleep(0.15)
        
        if cx < Config.FRAME_WIDTH // 2:
            right()
            print("[AVOIDANCE] Steering RIGHT")
        else:
            left()
            print("[AVOIDANCE] Steering LEFT")
        
        time.sleep(0.35)
        stop()
        self.state = "STOP"

# ----------------------------
# Main loop (maximum performance)
# ----------------------------
def main():
    print("[SYSTEM] Initializing high-performance ADAS...")
    
    # Start camera
    camera = ThreadedCamera(Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    
    # Initialize detectors
    obstacle_detector = UltraFastObstacleDetector()
    traffic_light_detector = OptimizedTrafficLightDetector()
    obstacle_tracker = FastObstacleTracker(Config.OBSTACLE_HISTORY_SIZE)
    motor_controller = SmartMotorController()
    
    print("[SYSTEM] ============================================")
    print("[SYSTEM] Ultra-Fast ADAS System Ready")
    print("[SYSTEM] Resolution: {}x{}".format(Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    print("[SYSTEM] Distance Estimation: ENABLED")
    print("[SYSTEM] Press 'q' to quit")
    print("[SYSTEM] ============================================")
    
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Get frame
            frame = camera.read()
            if frame is None:
                continue
            
            # Detect obstacles with distance
            obstacles, _ = obstacle_detector.detect(frame)
            obstacle_tracker.update(obstacles)
            stable_obstacles = obstacle_tracker.get_stable_obstacles()
            
            # Detect traffic lights
            traffic_light = traffic_light_detector.detect(frame)
            
            # Make decision
            motor_controller.decide(stable_obstacles, traffic_light)
            
            # Draw visualization
            display = frame.copy()
            
            # Draw obstacles with distance
            for i, obs in enumerate(stable_obstacles[:5]):  # Show top 5
                x, y, w, h = obs['bbox']
                distance = obs['distance']
                
                # Color by distance
                if distance < Config.CRITICAL_DISTANCE:
                    color = (0, 0, 255)  # Red
                elif distance < Config.WARNING_DISTANCE:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                
                # Distance label
                label = f"{distance}cm"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         0.5, 2)
                cv2.rectangle(display, (x, y-label_h-5), (x+label_w, y), color, -1)
                cv2.putText(display, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Traffic light indicator
            if traffic_light:
                tl_color = (0, 0, 255) if traffic_light == "RED" else \
                          (0, 165, 255) if traffic_light == "ORANGE" else (0, 255, 0)
                cv2.circle(display, (25, 25), 12, tl_color, -1)
                cv2.circle(display, (25, 25), 12, (255, 255, 255), 2)
            
            # ROI line
            roi_y = int(display.shape[0] * Config.ROI_TOP_RATIO)
            cv2.line(display, (0, roi_y), (Config.FRAME_WIDTH, roi_y), (255, 255, 0), 1)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Status overlay (compact)
            h = display.shape[0]
            status_bg = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
            cv2.putText(status_bg, f"FPS:{current_fps}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(status_bg, f"OBS:{len(stable_obstacles)}", (5, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(status_bg, f"{motor_controller.state}", (100, 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            if stable_obstacles:
                closest_dist = stable_obstacles[0]['distance']
                cv2.putText(status_bg, f"CLOSEST:{closest_dist}cm", (250, 24),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            display[-40:, :] = cv2.addWeighted(display[-40:, :], 0.3, status_bg, 0.7, 0)
            
            # Display
            cv2.imshow("ADAS - Ultra Fast", display)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        cv2.destroyAllWindows()
        stop()
        print("[SYSTEM] Shutdown complete")

if __name__ == "__main__":
    main()
