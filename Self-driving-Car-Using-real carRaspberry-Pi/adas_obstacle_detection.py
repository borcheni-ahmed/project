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
    FRAME_WIDTH = 480
    FRAME_HEIGHT = 360
    FPS = 30
    
    # Obstacle detection (STRICT)
    MIN_OBSTACLE_AREA = 1500  # Increased to filter noise
    MAX_OBSTACLE_AREA = 50000
    MIN_SOLIDITY = 0.3  # Object must be solid (not scattered pixels)
    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 4.0
    MIN_EXTENT = 0.3  # Portion of bounding box filled
    
    # Proximity thresholds
    CRITICAL_PROXIMITY = 75
    WARNING_PROXIMITY = 55
    SAFE_PROXIMITY = 35
    
    # ROI settings
    ROI_TOP_RATIO = 0.30
    ROI_BOTTOM_RATIO = 0.92
    ROI_LEFT_RATIO = 0.15  # Ignore edges
    ROI_RIGHT_RATIO = 0.85
    
    # Traffic light detection
    MIN_CIRCULARITY = 0.65
    MIN_LIGHT_AREA = 150
    MAX_LIGHT_AREA = 3000
    
    # Temporal filtering (STRICT)
    OBSTACLE_HISTORY_SIZE = 4  # Must appear in 4 frames
    MIN_DETECTIONS_REQUIRED = 3  # Must be detected 3/4 times
    LIGHT_HISTORY_SIZE = 3
    
    # Performance
    DETECTION_SCALE = 0.8
    
    # Edge rejection
    EDGE_MARGIN = 20  # Ignore objects too close to frame edges

# ----------------------------
# High-performance threaded camera
# ----------------------------
class ThreadedCamera:
    def __init__(self, width=480, height=360):
        self.picam2 = Picamera2()
        
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
            queue=False,
            controls={
                "AwbEnable": True,
                "AeEnable": True,
                "FrameRate": Config.FPS,
                "NoiseReductionMode": 1,  # Minimal noise reduction
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        self.frame = None
        self.lock = Lock()
        self.running = True
        
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        time.sleep(2)
    
    def _update(self):
        while self.running:
            try:
                new_frame = self.picam2.capture_array()
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
# Relative proximity estimation
# ----------------------------
class ProximityEstimator:
    
    @staticmethod
    def calculate_proximity(bbox, area, frame_height, frame_width):
        """Calculate proximity score (0-100, higher = closer/more dangerous)"""
        x, y, w, h = bbox
        
        # Factor 1: Object size (area)
        area_score = min(100, (area / Config.MAX_OBSTACLE_AREA) * 150)
        
        # Factor 2: Vertical position - lower in frame = closer
        roi_height = frame_height * (Config.ROI_BOTTOM_RATIO - Config.ROI_TOP_RATIO)
        roi_start = frame_height * Config.ROI_TOP_RATIO
        normalized_y = (y - roi_start) / roi_height
        position_score = (1 - normalized_y) * 100
        
        # Factor 3: Width
        width_score = min(100, (w / frame_width) * 200)
        
        # Weighted combination
        proximity = (area_score * 0.5) + (position_score * 0.3) + (width_score * 0.2)
        
        return max(0, min(100, int(proximity)))
    
    @staticmethod
    def proximity_to_level(proximity):
        """Convert proximity score to readable level"""
        if proximity >= 75:
            return "CRITICAL", (0, 0, 255)
        elif proximity >= 55:
            return "CLOSE", (0, 100, 255)
        elif proximity >= 35:
            return "MEDIUM", (0, 165, 255)
        elif proximity >= 20:
            return "FAR", (0, 255, 255)
        else:
            return "SAFE", (0, 255, 0)

# ----------------------------
# High-accuracy obstacle detector
# ----------------------------
class AccurateObstacleDetector:
    def __init__(self):
        # Background subtractor with conservative settings
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,  # Longer history for stability
            varThreshold=50,  # Higher threshold = less sensitive
            detectShadows=False
        )
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.frame_count = 0
        
    def detect(self, frame):
        """Accurate obstacle detection with strict filtering"""
        h, w = frame.shape[:2]
        
        # Define strict ROI (ignore edges and top)
        roi_top = int(h * Config.ROI_TOP_RATIO)
        roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
        roi_left = int(w * Config.ROI_LEFT_RATIO)
        roi_right = int(w * Config.ROI_RIGHT_RATIO)
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        roi_h, roi_w = roi.shape[:2]
        
        # Downscale for processing
        process_roi = cv2.resize(roi, None, fx=Config.DETECTION_SCALE, 
                                fy=Config.DETECTION_SCALE, 
                                interpolation=cv2.INTER_LINEAR)
        scale_factor = 1.0 / Config.DETECTION_SCALE
        
        # Convert to grayscale
        gray = cv2.cvtColor(process_roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Edge detection (for static objects)
        # Use bilateral filter to preserve edges while removing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 40, 120)
        
        # Method 2: Background subtraction (for moving objects)
        self.frame_count += 1
        if self.frame_count > 30:  # Only after background model is learned
            fg_mask = self.bg_subtractor.apply(process_roi, learningRate=0.001)
            _, motion = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
        else:
            motion = np.zeros_like(edges)
        
        # Method 3: Adaptive thresholding for contrast objects
        adaptive = cv2.adaptiveThreshold(filtered, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3)
        
        # Combine methods - require at least 2/3 to agree
        edge_motion = cv2.bitwise_and(edges, motion)
        edge_adaptive = cv2.bitwise_and(edges, adaptive)
        motion_adaptive = cv2.bitwise_and(motion, adaptive)
        
        # Combine overlaps
        combined = cv2.bitwise_or(edge_motion, edge_adaptive)
        combined = cv2.bitwise_or(combined, motion_adaptive)
        
        # Aggressive morphological cleanup
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel_medium, iterations=1)
        
        # Remove small blobs
        combined = cv2.erode(combined, self.kernel_small, iterations=1)
        combined = cv2.dilate(combined, self.kernel_medium, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Strict area filtering
            if area < Config.MIN_OBSTACLE_AREA or area > Config.MAX_OBSTACLE_AREA:
                continue
            
            # Get bounding box
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Calculate shape features for filtering
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # 1. Solidity check (convex hull vs actual area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < Config.MIN_SOLIDITY:
                continue  # Too scattered/noisy
            
            # 2. Aspect ratio check
            aspect = w_rect / float(h_rect) if h_rect > 0 else 0
            if aspect < Config.MIN_ASPECT_RATIO or aspect > Config.MAX_ASPECT_RATIO:
                continue
            
            # 3. Extent check (area vs bounding box area)
            bbox_area = w_rect * h_rect
            extent = area / bbox_area if bbox_area > 0 else 0
            
            if extent < Config.MIN_EXTENT:
                continue  # Too sparse
            
            # 4. Edge rejection - ignore objects at frame edges
            if (x < Config.EDGE_MARGIN or 
                y < Config.EDGE_MARGIN or 
                x + w_rect > process_roi.shape[1] - Config.EDGE_MARGIN or
                y + h_rect > process_roi.shape[0] - Config.EDGE_MARGIN):
                continue
            
            # Scale back to original coordinates
            x_orig = int(x * scale_factor) + roi_left
            y_orig = int(y * scale_factor) + roi_top
            w_orig = int(w_rect * scale_factor)
            h_orig = int(h_rect * scale_factor)
            
            # Calculate proximity
            proximity = ProximityEstimator.calculate_proximity(
                (x_orig, y_orig, w_orig, h_orig),
                area * (scale_factor ** 2),
                h, w
            )
            
            # Calculate center
            cx = x_orig + w_orig // 2
            cy = y_orig + h_orig // 2
            
            obstacles.append({
                'bbox': (x_orig, y_orig, w_orig, h_orig),
                'area': area * (scale_factor ** 2),
                'center': (cx, cy),
                'proximity': proximity,
                'solidity': solidity,
                'extent': extent,
                'criticality': proximity
            })
        
        # Sort by proximity
        obstacles.sort(key=lambda x: x['proximity'], reverse=True)
        
        return obstacles[:6], combined

# ----------------------------
# Traffic light detector
# ----------------------------
class OptimizedTrafficLightDetector:
    def __init__(self):
        self.light_history = deque(maxlen=Config.LIGHT_HISTORY_SIZE)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect(self, frame):
        """Detect traffic lights"""
        h, w = frame.shape[:2]
        roi = frame[0:h//3, w//3:2*w//3]
        roi_small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
        
        scores = {
            "RED": self._quick_detect(hsv, "RED"),
            "ORANGE": self._quick_detect(hsv, "ORANGE"),
            "GREEN": self._quick_detect(hsv, "GREEN")
        }
        
        max_score = max(scores.values())
        detected = None
        
        if max_score > 150:  # Higher threshold
            detected = max(scores, key=scores.get)
        
        self.light_history.append(detected)
        
        if len(self.light_history) >= 2:
            recent = list(self.light_history)[-2:]
            non_none = [x for x in recent if x is not None]
            if len(non_none) >= 2 and non_none[0] == non_none[1]:
                return non_none[0]
        
        return None
    
    def _quick_detect(self, hsv, color):
        if color == "RED":
            mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "ORANGE":
            mask = cv2.inRange(hsv, (10, 100, 100), (25, 255, 255))
        else:
            mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.countNonZero(mask)

# ----------------------------
# Strict temporal obstacle tracker
# ----------------------------
class StrictObstacleTracker:
    def __init__(self, history_size=4):
        self.history = deque(maxlen=history_size)
        self.obstacle_tracks = {}  # Track obstacles by position
        self.next_id = 0
        
    def update(self, obstacles):
        """Track obstacles over time with position matching"""
        self.history.append(obstacles)
    
    def get_stable_obstacles(self):
        """Return only obstacles that appear consistently"""
        if len(self.history) < Config.MIN_DETECTIONS_REQUIRED:
            return []
        
        # Count how many times each obstacle position appears
        position_map = {}
        
        for frame_obstacles in self.history:
            for obs in frame_obstacles:
                cx, cy = obs['center']
                # Round position to grid (allow small movement)
                grid_x = round(cx / 30) * 30
                grid_y = round(cy / 30) * 30
                key = (grid_x, grid_y)
                
                if key not in position_map:
                    position_map[key] = []
                position_map[key].append(obs)
        
        # Keep only obstacles detected MIN_DETECTIONS_REQUIRED times
        stable_obstacles = []
        for key, detections in position_map.items():
            if len(detections) >= Config.MIN_DETECTIONS_REQUIRED:
                # Use most recent detection
                stable_obstacles.append(detections[-1])
        
        return stable_obstacles

# ----------------------------
# Motor controller
# ----------------------------
class SmartMotorController:
    def __init__(self):
        self.state = "STOP"
        self.last_avoidance_time = 0
        
    def decide(self, obstacles, traffic_light):
        current_time = time.time()
        
        if traffic_light == "RED":
            self._execute_stop()
            return
        
        if obstacles:
            closest = obstacles[0]
            proximity = closest['proximity']
            
            if proximity >= Config.CRITICAL_PROXIMITY:
                if current_time - self.last_avoidance_time > 2.0:
                    self._execute_avoidance(closest)
                    self.last_avoidance_time = current_time
                return
            
            elif proximity >= Config.WARNING_PROXIMITY:
                self._execute_stop()
                return
        
        if traffic_light == "ORANGE":
            self._execute_stop()
            return
        
        if traffic_light == "GREEN" or traffic_light is None:
            if not obstacles or obstacles[0]['proximity'] < Config.SAFE_PROXIMITY:
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
        proximity = obstacle['proximity']
        
        print(f"[AVOIDANCE] Proximity: {proximity}%")
        
        stop()
        time.sleep(0.2)
        
        if cx < Config.FRAME_WIDTH // 2:
            right()
            print("[AVOIDANCE] RIGHT")
        else:
            left()
            print("[AVOIDANCE] LEFT")
        
        time.sleep(0.4)
        stop()
        self.state = "STOP"

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("[SYSTEM] Initializing Accurate ADAS...")
    
    camera = ThreadedCamera(Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    
    obstacle_detector = AccurateObstacleDetector()
    traffic_light_detector = OptimizedTrafficLightDetector()
    obstacle_tracker = StrictObstacleTracker(Config.OBSTACLE_HISTORY_SIZE)
    motor_controller = SmartMotorController()
    
    print("[SYSTEM] ============================================")
    print("[SYSTEM] High-Accuracy ADAS System")
    print("[SYSTEM] Strict filtering enabled")
    print("[SYSTEM] Temporal tracking: Must appear in 3/4 frames")
    print("[SYSTEM] Press 'q' to quit, 'd' for debug view")
    print("[SYSTEM] ============================================")
    
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    show_debug = False
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            # Detect obstacles
            obstacles, debug_mask = obstacle_detector.detect(frame)
            obstacle_tracker.update(obstacles)
            stable_obstacles = obstacle_tracker.get_stable_obstacles()
            
            # Detect traffic lights
            traffic_light = traffic_light_detector.detect(frame)
            
            # Make decision
            motor_controller.decide(stable_obstacles, traffic_light)
            
            # Visualization
            display = frame.copy()
            
            # Draw ROI boundary
            h, w = display.shape[:2]
            roi_top = int(h * Config.ROI_TOP_RATIO)
            roi_left = int(w * Config.ROI_LEFT_RATIO)
            roi_right = int(w * Config.ROI_RIGHT_RATIO)
            cv2.rectangle(display, (roi_left, roi_top), 
                         (roi_right, int(h * Config.ROI_BOTTOM_RATIO)), 
                         (255, 255, 0), 1)
            
            # Draw stable obstacles only
            for obs in stable_obstacles[:5]:
                x, y, w_rect, h_rect = obs['bbox']
                proximity = obs['proximity']
                
                level, color = ProximityEstimator.proximity_to_level(proximity)
                
                cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), color, 2)
                
                # Label
                label = f"{proximity}% {level}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(display, (x, y-label_size[1]-5), 
                             (x+label_size[0]+4, y), color, -1)
                cv2.putText(display, label, (x+2, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw quality metrics (optional)
                metrics = f"S:{obs['solidity']:.2f} E:{obs['extent']:.2f}"
                cv2.putText(display, metrics, (x, y+h_rect+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Traffic light
            if traffic_light:
                tl_color = (0, 0, 255) if traffic_light == "RED" else \
                          (0, 165, 255) if traffic_light == "ORANGE" else (0, 255, 0)
                cv2.circle(display, (25, 25), 12, tl_color, -1)
                cv2.circle(display, (25, 25), 12, (255, 255, 255), 2)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Status
            status_bg = np.zeros((50, w, 3), dtype=np.uint8)
            cv2.putText(status_bg, f"FPS:{current_fps}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(status_bg, f"Detected:{len(obstacles)} Stable:{len(stable_obstacles)}", 
                       (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(status_bg, f"{motor_controller.state}", (150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            if stable_obstacles:
                closest = stable_obstacles[0]
                level, prox_color = ProximityEstimator.proximity_to_level(closest['proximity'])
                cv2.putText(status_bg, f"{level}:{closest['proximity']}%", (280, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, prox_color, 1)
            
            display[-50:, :] = cv2.addWeighted(display[-50:, :], 0.3, status_bg, 0.7, 0)
            
            cv2.imshow("ADAS - Accurate Detection", display)
            
            # Debug view
            if show_debug and debug_mask is not None:
                debug_resized = cv2.resize(debug_mask, (320, 240))
                cv2.imshow("Debug - Detection Mask", debug_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Debug - Detection Mask")
            
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
