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
    # Camera settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Obstacle detection
    MIN_OBSTACLE_AREA = 1000
    MAX_OBSTACLE_AREA = 80000
    CRITICAL_DISTANCE_THRESHOLD = 25000
    SAFE_DISTANCE_THRESHOLD = 12000
    
    # ROI settings
    ROI_TOP_RATIO = 0.35
    ROI_BOTTOM_RATIO = 0.95
    
    # Traffic light detection (strict)
    MIN_CIRCULARITY = 0.65
    MIN_LIGHT_AREA = 150
    MAX_LIGHT_AREA = 4000
    
    # History for stability
    OBSTACLE_HISTORY_SIZE = 3
    LIGHT_HISTORY_SIZE = 3
    
    # Performance
    SKIP_FRAMES = 1  # Process every N frames

# ----------------------------
# Threaded camera capture for better performance
# ----------------------------
class ThreadedCamera:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        
        # Optimized configuration
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,  # Reduce buffer for less latency
            controls={
                "AwbEnable": True,
                "AeEnable": True,
                "FrameRate": Config.FPS,
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        self.frame = None
        self.lock = Lock()
        self.running = True
        
        # Start capture thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Warm up
        time.sleep(2)
    
    def _update(self):
        """Continuously capture frames in background"""
        while self.running:
            try:
                new_frame = self.picam2.capture_array()
                with self.lock:
                    # Convert RGB to BGR (OpenCV format) - THIS FIXES COLOR SWAP!
                    self.frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[CAMERA ERROR] {e}")
                time.sleep(0.1)
    
    def read(self):
        """Get latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def release(self):
        """Stop camera"""
        self.running = False
        self.thread.join()
        self.picam2.stop()

# ----------------------------
# Fast color correction
# ----------------------------
def correct_colors_fast(frame):
    """Lightweight color correction"""
    # Simple brightness and contrast adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Fast CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return corrected

# ----------------------------
# Optimized obstacle detection
# ----------------------------
class FastObstacleDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self.frame_count = 0
        
    def detect(self, frame):
        """Lightweight obstacle detection"""
        self.frame_count += 1
        
        h, w = frame.shape[:2]
        roi_top = int(h * Config.ROI_TOP_RATIO)
        roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
        
        roi = frame[roi_top:roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Simple edge detection (fast)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        
        # Method 2: Background subtraction
        fg_mask = self.bg_subtractor.apply(roi, learningRate=0.005)
        _, motion = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Combine
        combined = cv2.bitwise_or(edges, motion)
        
        # Quick morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_OBSTACLE_AREA < area < Config.MAX_OBSTACLE_AREA:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Quick aspect ratio check
                aspect = w_rect / float(h_rect) if h_rect > 0 else 0
                if 0.2 < aspect < 5.0:
                    cx = x + w_rect // 2
                    cy = y + h_rect // 2
                    
                    # Simple criticality
                    area_score = min(100, (area / Config.CRITICAL_DISTANCE_THRESHOLD) * 100)
                    position_score = ((roi.shape[0] - cy) / roi.shape[0]) * 100
                    criticality = (area_score * 0.7) + (position_score * 0.3)
                    
                    obstacles.append({
                        'bbox': (x, y + roi_top, w_rect, h_rect),
                        'area': area,
                        'center': (cx, cy + roi_top),
                        'criticality': criticality
                    })
        
        obstacles.sort(key=lambda x: x['criticality'], reverse=True)
        return obstacles[:5]  # Return top 5 only

# ----------------------------
# Fixed traffic light detection
# ----------------------------
class FastTrafficLightDetector:
    def __init__(self):
        self.light_history = deque(maxlen=Config.LIGHT_HISTORY_SIZE)
        
    def detect(self, frame):
        """Detect traffic lights with CORRECT BGR colors"""
        h, w = frame.shape[:2]
        roi = frame[0:h//2, w//4:3*w//4]  # Focus on center-top
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Detect each color
        red_score = self._detect_color(hsv, "RED")
        orange_score = self._detect_color(hsv, "ORANGE")
        green_score = self._detect_color(hsv, "GREEN")
        
        # Determine which light is active
        detected_state = None
        max_score = max(red_score, orange_score, green_score)
        
        if max_score > 200:  # Minimum pixels threshold
            if red_score == max_score:
                detected_state = "RED"
            elif orange_score == max_score:
                detected_state = "ORANGE"
            elif green_score == max_score:
                detected_state = "GREEN"
        
        # Add to history
        self.light_history.append(detected_state)
        
        # Return only if consistent
        if len(self.light_history) >= 2:
            non_none = [s for s in list(self.light_history)[-2:] if s is not None]
            if len(non_none) >= 2 and non_none[0] == non_none[1]:
                return non_none[0]
        
        return None
    
    def _detect_color(self, hsv, color):
        """Detect specific color and check circularity"""
        if color == "RED":
            # Red wraps around in HSV (0-10 and 170-180)
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "ORANGE":
            # Orange/Yellow range
            mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
        else:  # GREEN
            mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find circular contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_LIGHT_AREA < area < Config.MAX_LIGHT_AREA:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > Config.MIN_CIRCULARITY:
                        total_score += area
        
        return total_score

# ----------------------------
# Obstacle tracker
# ----------------------------
class ObstacleTracker:
    def __init__(self, history_size=3):
        self.history = deque(maxlen=history_size)
    
    def update(self, obstacles):
        self.history.append(obstacles)
    
    def get_stable_obstacles(self):
        if len(self.history) < 2:
            return []
        
        if all(len(obs) > 0 for obs in self.history):
            return self.history[-1]
        
        return []

# ----------------------------
# Motor controller
# ----------------------------
class MotorController:
    def __init__(self):
        self.state = "STOP"
        self.avoidance_count = 0
        self.last_avoidance_time = 0
        
    def decide(self, obstacles, traffic_light):
        current_time = time.time()
        
        # Red light = stop
        if traffic_light == "RED":
            self._execute_stop()
            return
        
        # Critical obstacle
        if obstacles:
            most_critical = obstacles[0]
            
            if most_critical['criticality'] > 60:
                if current_time - self.last_avoidance_time > 2.0:
                    self._execute_avoidance(most_critical)
                    self.last_avoidance_time = current_time
                return
            
            elif most_critical['criticality'] > 35:
                self._execute_stop()
                return
        
        # Orange light
        if traffic_light == "ORANGE":
            self._execute_stop()
            return
        
        # Green or clear - go
        if traffic_light == "GREEN" or traffic_light is None:
            if not obstacles:
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
        frame_center = Config.FRAME_WIDTH // 2
        
        print(f"[AVOIDANCE] Criticality: {obstacle['criticality']:.1f}")
        
        stop()
        time.sleep(0.2)
        
        if cx < frame_center:
            right()
            print("[AVOIDANCE] Steering RIGHT")
        else:
            left()
            print("[AVOIDANCE] Steering LEFT")
        
        time.sleep(0.4)
        stop()
        self.state = "STOP"

# ----------------------------
# Main function (optimized)
# ----------------------------
def main():
    # Initialize threaded camera
    print("[SYSTEM] Starting threaded camera...")
    camera = ThreadedCamera(Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    
    # Initialize detectors
    obstacle_detector = FastObstacleDetector()
    traffic_light_detector = FastTrafficLightDetector()
    obstacle_tracker = ObstacleTracker(Config.OBSTACLE_HISTORY_SIZE)
    motor_controller = MotorController()
    
    print("[SYSTEM] ============================================")
    print("[SYSTEM] ADAS System Started (Optimized)")
    print("[SYSTEM] BGR Color Fix Applied")
    print("[SYSTEM] Press 'q' to quit, 's' to save frame")
    print("[SYSTEM] ============================================")
    
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            # Get frame from threaded camera
            frame = camera.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Process every N frames for better performance
            if frame_count % (Config.SKIP_FRAMES + 1) != 0:
                continue
            
            # Light color correction
            corrected_frame = correct_colors_fast(frame)
            
            # Detect obstacles
            obstacles = obstacle_detector.detect(corrected_frame)
            obstacle_tracker.update(obstacles)
            stable_obstacles = obstacle_tracker.get_stable_obstacles()
            
            # Detect traffic lights
            traffic_light = traffic_light_detector.detect(corrected_frame)
            
            # Draw obstacles
            display_frame = corrected_frame.copy()
            for obs in stable_obstacles:
                x, y, w, h = obs['bbox']
                criticality = obs['criticality']
                
                # Color based on criticality
                if criticality > 60:
                    color = (0, 0, 255)  # Red
                elif criticality > 35:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, f"{criticality:.0f}%", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw traffic light
            if traffic_light:
                tl_color = (0, 0, 255) if traffic_light == "RED" else \
                          (0, 165, 255) if traffic_light == "ORANGE" else (0, 255, 0)
                cv2.circle(display_frame, (30, 30), 15, tl_color, -1)
                cv2.putText(display_frame, traffic_light, (55, 38),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_color, 2)
            
            # Draw ROI
            h = display_frame.shape[0]
            roi_top = int(h * Config.ROI_TOP_RATIO)
            cv2.line(display_frame, (0, roi_top), (Config.FRAME_WIDTH, roi_top), 
                    (255, 0, 0), 1)
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Status overlay
            cv2.putText(display_frame, f"FPS: {current_fps}", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Obstacles: {len(stable_obstacles)}", (10, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"State: {motor_controller.state}", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Make decision
            motor_controller.decide(stable_obstacles, traffic_light)
            
            # Display
            cv2.imshow("ADAS - Optimized View", display_frame)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] Shutting down...")
                break
            elif key == ord('s'):
                filename = f"adas_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[SYSTEM] Saved: {filename}")
            
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
        print("[SYSTEM] Cleanup complete")

if __name__ == "__main__":
    main()
