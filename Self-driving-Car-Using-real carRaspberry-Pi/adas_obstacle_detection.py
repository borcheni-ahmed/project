from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque

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
    
    # Color correction
    BRIGHTNESS_ADJUST = 30
    CONTRAST_ADJUST = 1.3
    SATURATION_ADJUST = 1.2
    
    # Obstacle detection
    MIN_OBSTACLE_AREA = 1200
    MAX_OBSTACLE_AREA = 100000
    CRITICAL_DISTANCE_THRESHOLD = 30000  # Area-based
    SAFE_DISTANCE_THRESHOLD = 15000
    
    # ROI settings
    ROI_TOP_RATIO = 0.35
    ROI_BOTTOM_RATIO = 0.95
    
    # Motion detection sensitivity
    MOTION_THRESHOLD = 25
    MIN_MOTION_AREA = 500
    
    # Traffic light detection (strict)
    MIN_CIRCULARITY = 0.7  # To detect circular lights
    MIN_LIGHT_AREA = 200
    MAX_LIGHT_AREA = 5000
    VERTICAL_ARRANGEMENT_TOLERANCE = 50  # pixels
    
    # History for stability
    OBSTACLE_HISTORY_SIZE = 3
    LIGHT_HISTORY_SIZE = 4

# ----------------------------
# Camera color correction
# ----------------------------
def correct_colors(frame):
    """Apply color correction to improve camera output"""
    # Convert to LAB color space for better color correction
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Additional adjustments
    corrected = cv2.convertScaleAbs(corrected, alpha=Config.CONTRAST_ADJUST, 
                                    beta=Config.BRIGHTNESS_ADJUST)
    
    # Increase saturation
    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=Config.SATURATION_ADJUST, beta=0)
    hsv = cv2.merge([h, s, v])
    corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return corrected

# ----------------------------
# Precise obstacle detection
# ----------------------------
class ObstacleDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.prev_gray = None
        
    def detect(self, frame):
        """Detect obstacles using multiple methods"""
        h, w = frame.shape[:2]
        roi_top = int(h * Config.ROI_TOP_RATIO)
        roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
        
        # Focus on road area
        roi = frame[roi_top:roi_bottom, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Edge detection
        edges = self._detect_edges(gray_roi)
        
        # Method 2: Background subtraction (motion-based)
        motion_mask = self._detect_motion(roi)
        
        # Combine both methods
        combined = cv2.bitwise_or(edges, motion_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_OBSTACLE_AREA < area < Config.MAX_OBSTACLE_AREA:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (reject very wide/thin objects)
                aspect_ratio = w_rect / float(h_rect) if h_rect > 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    # Calculate center position
                    cx = x + w_rect // 2
                    cy = y + h_rect // 2
                    
                    # Obstacle criticality based on area and position
                    criticality = self._calculate_criticality(area, cy, roi.shape[0])
                    
                    obstacles.append({
                        'bbox': (x, y + roi_top, w_rect, h_rect),
                        'area': area,
                        'center': (cx, cy + roi_top),
                        'criticality': criticality
                    })
        
        # Sort by criticality (most critical first)
        obstacles.sort(key=lambda x: x['criticality'], reverse=True)
        
        return obstacles, combined
    
    def _detect_edges(self, gray):
        """Detect edges with adaptive thresholding"""
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Canny edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Combine both
        combined = cv2.bitwise_or(thresh, edges)
        
        return combined
    
    def _detect_motion(self, roi):
        """Detect moving obstacles"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(roi, learningRate=0.01)
        
        # Threshold to get binary mask
        _, motion_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        return motion_mask
    
    def _calculate_criticality(self, area, y_pos, roi_height):
        """Calculate how critical an obstacle is (0-100)"""
        # Larger objects are more critical
        area_score = min(100, (area / Config.CRITICAL_DISTANCE_THRESHOLD) * 100)
        
        # Lower position (closer to car) is more critical
        position_score = ((roi_height - y_pos) / roi_height) * 100
        
        # Weighted combination
        criticality = (area_score * 0.7) + (position_score * 0.3)
        
        return criticality

# ----------------------------
# Traffic light detection (strict)
# ----------------------------
class TrafficLightDetector:
    def __init__(self):
        self.light_history = deque(maxlen=Config.LIGHT_HISTORY_SIZE)
        
    def detect(self, frame):
        """Detect traffic lights only if they look like actual traffic lights"""
        h, w = frame.shape[:2]
        
        # Focus on upper portion where traffic lights are
        roi = frame[0:h//2, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Detect colored circles (traffic lights are circular)
        red_lights = self._find_circular_lights(hsv, "RED")
        orange_lights = self._find_circular_lights(hsv, "ORANGE")
        green_lights = self._find_circular_lights(hsv, "GREEN")
        
        # Check if lights are vertically arranged (typical traffic light pattern)
        detected_state = None
        if red_lights or orange_lights or green_lights:
            if self._is_traffic_light_arrangement(red_lights, orange_lights, green_lights):
                if red_lights:
                    detected_state = "RED"
                elif orange_lights:
                    detected_state = "ORANGE"
                elif green_lights:
                    detected_state = "GREEN"
        
        # Add to history for stability
        self.light_history.append(detected_state)
        
        # Return only if detected consistently
        if len(self.light_history) >= 3:
            non_none = [s for s in self.light_history if s is not None]
            if len(non_none) >= 2 and len(set(non_none)) == 1:
                return non_none[0]
        
        return None
    
    def _find_circular_lights(self, hsv, color):
        """Find circular colored objects (potential traffic lights)"""
        if color == "RED":
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "ORANGE":
            mask = cv2.inRange(hsv, np.array([8, 100, 100]), np.array([25, 255, 255]))
        else:  # GREEN
            mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lights = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if Config.MIN_LIGHT_AREA < area < Config.MAX_LIGHT_AREA:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > Config.MIN_CIRCULARITY:
                    x, y, w, h = cv2.boundingRect(contour)
                    lights.append({'bbox': (x, y, w, h), 'area': area})
        
        return lights
    
    def _is_traffic_light_arrangement(self, red, orange, green):
        """Check if detected lights are arranged like a traffic light"""
        all_lights = red + orange + green
        
        if len(all_lights) < 1:
            return False
        
        # If only one light detected, check if it's reasonably sized
        if len(all_lights) == 1:
            return True
        
        # If multiple lights, check vertical arrangement
        centers_y = [bbox[1] + bbox[3]//2 for _, bbox in 
                     [(l, l['bbox']) for l in all_lights]]
        
        # Check if there's vertical spacing between lights
        centers_y.sort()
        for i in range(len(centers_y) - 1):
            spacing = centers_y[i+1] - centers_y[i]
            if Config.VERTICAL_ARRANGEMENT_TOLERANCE < spacing < 200:
                return True
        
        return False

# ----------------------------
# Obstacle tracking with history
# ----------------------------
class ObstacleTracker:
    def __init__(self, history_size=3):
        self.history = deque(maxlen=history_size)
    
    def update(self, obstacles):
        self.history.append(obstacles)
    
    def get_stable_obstacles(self):
        """Return obstacles that appear consistently"""
        if len(self.history) < 2:
            return []
        
        # Return most recent obstacles if they're consistent
        if all(len(obs) > 0 for obs in self.history):
            return self.history[-1]
        
        return []

# ----------------------------
# Motor controller with precise logic
# ----------------------------
class MotorController:
    def __init__(self):
        self.state = "STOP"
        self.avoidance_count = 0
        self.last_avoidance_time = 0
        
    def decide(self, obstacles, traffic_light):
        """Make driving decision based on inputs"""
        current_time = time.time()
        
        # Priority 1: Red traffic light
        if traffic_light == "RED":
            self._execute_stop()
            return
        
        # Priority 2: Critical obstacle
        if obstacles and len(obstacles) > 0:
            most_critical = obstacles[0]
            
            if most_critical['criticality'] > 60:
                # Very close obstacle - avoid
                if current_time - self.last_avoidance_time > 2.0:
                    self._execute_avoidance(most_critical)
                    self.last_avoidance_time = current_time
                return
            
            elif most_critical['criticality'] > 35:
                # Medium distance - slow down/stop
                self._execute_stop()
                return
        
        # Priority 3: Orange light - prepare to stop
        if traffic_light == "ORANGE":
            self._execute_stop()
            return
        
        # Priority 4: Green light or clear road - go forward
        if traffic_light == "GREEN" or traffic_light is None:
            if not obstacles or len(obstacles) == 0:
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
        """Smart avoidance based on obstacle position"""
        cx = obstacle['center'][0]
        frame_center = Config.FRAME_WIDTH // 2
        
        print(f"[AVOIDANCE] Obstacle criticality: {obstacle['criticality']:.1f}")
        
        stop()
        time.sleep(0.2)
        
        # Steer away from obstacle position
        if cx < frame_center:
            # Obstacle on left, go right
            right()
            print("[AVOIDANCE] Obstacle on left, steering RIGHT")
        else:
            # Obstacle on right, go left
            left()
            print("[AVOIDANCE] Obstacle on right, steering LEFT")
        
        time.sleep(0.4)
        stop()
        self.state = "STOP"
        self.avoidance_count += 1

# ----------------------------
# Main function
# ----------------------------
def main():
    # Initialize camera with manual settings for better color
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_preview_configuration(
        main={"size": (Config.FRAME_WIDTH, Config.FRAME_HEIGHT), "format": "RGB888"},
        controls={
            "AwbEnable": True,  # Auto white balance
            "AeEnable": True,   # Auto exposure
            "Brightness": 0.0,
            "Contrast": 1.0,
            "Saturation": 1.2
        }
    )
    picam2.configure(config)
    picam2.start()
    
    # Warm-up period
    print("[SYSTEM] Camera warming up...")
    time.sleep(3)
    
    # Initialize components
    obstacle_detector = ObstacleDetector()
    traffic_light_detector = TrafficLightDetector()
    obstacle_tracker = ObstacleTracker(Config.OBSTACLE_HISTORY_SIZE)
    motor_controller = MotorController()
    
    print("[SYSTEM] ============================================")
    print("[SYSTEM] ADAS System Started")
    print("[SYSTEM] Press 'q' to quit, 's' to save frame")
    print("[SYSTEM] ============================================")
    
    frame_count = 0
    
    try:
        while True:
            # Capture and correct frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            corrected_frame = correct_colors(frame)
            
            # Detect obstacles
            obstacles, debug_mask = obstacle_detector.detect(corrected_frame)
            obstacle_tracker.update(obstacles)
            stable_obstacles = obstacle_tracker.get_stable_obstacles()
            
            # Detect traffic lights
            traffic_light = traffic_light_detector.detect(corrected_frame)
            
            # Draw obstacles
            display_frame = corrected_frame.copy()
            for i, obs in enumerate(stable_obstacles):
                x, y, w, h = obs['bbox']
                criticality = obs['criticality']
                
                # Color based on criticality
                if criticality > 60:
                    color = (0, 0, 255)  # Red - critical
                elif criticality > 35:
                    color = (0, 165, 255)  # Orange - warning
                else:
                    color = (0, 255, 255)  # Yellow - detected
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, f"C:{criticality:.0f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw traffic light status
            if traffic_light:
                tl_color = (0, 0, 255) if traffic_light == "RED" else \
                          (0, 165, 255) if traffic_light == "ORANGE" else (0, 255, 0)
                cv2.putText(display_frame, f"TRAFFIC: {traffic_light}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_color, 2)
            
            # Draw ROI lines
            h = display_frame.shape[0]
            roi_top = int(h * Config.ROI_TOP_RATIO)
            roi_bottom = int(h * Config.ROI_BOTTOM_RATIO)
            cv2.line(display_frame, (0, roi_top), (Config.FRAME_WIDTH, roi_top), 
                    (255, 0, 0), 1)
            cv2.line(display_frame, (0, roi_bottom), (Config.FRAME_WIDTH, roi_bottom), 
                    (255, 0, 0), 1)
            
            # Status info
            obstacle_count = len(stable_obstacles)
            cv2.putText(display_frame, f"Obstacles: {obstacle_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"State: {motor_controller.state}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Make driving decision
            motor_controller.decide(stable_obstacles, traffic_light)
            
            # Display frames
            cv2.imshow("ADAS - Main View", display_frame)
            
            # Show debug mask (smaller window)
            if debug_mask is not None:
                debug_resized = cv2.resize(debug_mask, (320, 240))
                cv2.imshow("ADAS - Detection Mask", debug_resized)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] Shutting down...")
                break
            elif key == ord('s'):
                cv2.imwrite(f"adas_frame_{frame_count}.jpg", display_frame)
                print(f"[SYSTEM] Frame saved: adas_frame_{frame_count}.jpg")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        stop()
        print("[SYSTEM] Cleanup complete")

if __name__ == "__main__":
    main()
