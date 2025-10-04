# Moss-Stress-Detection-with-Live-HSV-Values
import cv2
import numpy as np
import requests
import imutils
import time
from datetime import datetime

def detect_colors_with_live_hsv(use_esp32_cam=True, esp32_url='http://10.15.75.31/cam-hi.jpg'):
    if use_esp32_cam:
        print(f"Connecting to ESP32-CAM at: {esp32_url}")
        try:
            response = requests.get(esp32_url, timeout=5)
            if response.status_code != 200:
                print("Error: Could not connect to ESP32-CAM stream")
                return
        except requests.exceptions.RequestException:
            print("Error: Could not reach ESP32-CAM. Check IP address and network connection.")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access local camera")
            return

    color_ranges = {
        'healthy_green': ([40, 40, 40], [90, 255, 255]),
        'pale_green': ([35, 30, 80], [50, 120, 200]),
        'chlorotic_yellow': ([25, 50, 50], [40, 255, 255]),
        'necrotic_brown': ([5, 10, 10], [20, 180, 120])
    }

    # Use the same green color for all bounding boxes - matching your screenshot
    uniform_display_color = (0, 255, 0)  # Bright green for all detections

    # Keep different colors for the text overlay percentages at bottom
    display_colors = {
        'healthy_green': (0, 255, 0),      
        'pale_green': (128, 255, 128),      
        'chlorotic_yellow': (0, 255, 255),  
        'necrotic_brown': (0, 165, 255)     
    }

    zoom_factor = 1.0
    zoom_step = 0.2
    pan_x, pan_y = 0, 0
    pan_step = 20
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("="*80)
    print("MOSS STRESS DETECTION WITH LIVE HSV VALUES")
    print("="*80)
    print(f"Session started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("Controls:")
    print("  '+' or '=' : Zoom in")
    print("  '-' or '_' : Zoom out")
    print("  'r' : Reset zoom and pan")
    print("  Arrow keys : Pan when zoomed in")
    print("  'q' : Quit")
    print("="*80)
    print()

    frame_count = 0
    last_print_time = time.time()
    print_interval = 1.0  # Print HSV values every 1 second

    while True:
        current_time = time.time()
        current_datetime = datetime.now()
        elapsed_time = current_time - start_time
        
        if use_esp32_cam:
            try:
                img_resp = requests.get(esp32_url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)
                frame = imutils.resize(frame, width=640)
            except Exception as e:
                print(f"[{current_datetime.strftime('%H:%M:%S')}] Failed to grab frame:", e)
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                print(f"[{current_datetime.strftime('%H:%M:%S')}] Failed to grab frame")
                break

        # Apply digital zoom
        if zoom_factor > 1.0:
            h, w = frame.shape[:2]
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            center_x = w // 2 + pan_x
            center_y = h // 2 + pan_y
            x1 = max(0, min(center_x - crop_w // 2, w - crop_w))
            y1 = max(0, min(center_y - crop_h // 2, h - crop_h))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            cropped = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        total_pixels = frame.shape[0] * frame.shape[1]
        stress_pixels = {key: 0 for key in color_ranges}
        detected_hsv_values = {key: [] for key in color_ranges}
        percentages = {}

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            count = cv2.countNonZero(mask)
            stress_pixels[color_name] = count
            percentages[color_name] = (count / total_pixels) * 100

            # Extract actual HSV values from detected regions
            if count > 0:
                hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
                non_zero_pixels = hsv_masked[mask > 0]
                if len(non_zero_pixels) > 0:
                    # Sample some pixels to get representative HSV values
                    sample_size = min(50, len(non_zero_pixels))
                    sampled_pixels = non_zero_pixels[::len(non_zero_pixels)//sample_size]
                    detected_hsv_values[color_name] = sampled_pixels.tolist()

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Use uniform green color for all bounding boxes
                    cv2.rectangle(frame, (x, y), (x + w, y + h), uniform_display_color, 2)
                    cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, uniform_display_color, 2)

        # Print live HSV values every interval
        if current_time - last_print_time >= print_interval:
            # Format elapsed time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)
            elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
            
            print(f"\n[FRAME {frame_count}] LIVE HSV VALUES")
            print(f"Current Time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed Time: {elapsed_str}")
            print(f"FPS: {frame_count / elapsed_time:.1f}")
            print("-" * 80)
            
            for color_name in color_ranges:
                pixel_count = stress_pixels[color_name]
                percentage = percentages[color_name]
                
                print(f"{color_name.upper()}:")
                print(f"  Pixels detected: {pixel_count} ({percentage:.2f}%)")
                
                if detected_hsv_values[color_name]:
                    hsv_array = np.array(detected_hsv_values[color_name])
                    mean_hsv = np.mean(hsv_array, axis=0)
                    std_hsv = np.std(hsv_array, axis=0)
                    min_hsv = np.min(hsv_array, axis=0)
                    max_hsv = np.max(hsv_array, axis=0)
                    
                    print(f"  Mean HSV: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
                    print(f"  Std Dev:  H={std_hsv[0]:.1f}, S={std_hsv[1]:.1f}, V={std_hsv[2]:.1f}")
                    print(f"  Range:    H={min_hsv[0]}-{max_hsv[0]}, S={min_hsv[1]}-{max_hsv[1]}, V={min_hsv[2]}-{max_hsv[2]}")
                else:
                    print("  No pixels detected in this range")
                print()
            
            last_print_time = current_time

        # Add overlay information to the frame
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        # Create semi-transparent background for text overlay
        overlay = frame.copy()
        
        # Top section for zoom and time
        cv2.rectangle(overlay, (0, 0), (frame_width, 85), (0, 0, 0), -1)
        
        # Bottom section for percentages
        bottom_height = 140
        cv2.rectangle(overlay, (0, frame_height - bottom_height), (frame_width, frame_height), (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Zoom text (top left)
        zoom_text = f"Zoom: {zoom_factor:.1f}x"
        cv2.putText(frame, zoom_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current time display (below zoom, prominent)
        time_text = current_datetime.strftime('%H:%M:%S')
        cv2.putText(frame, f"Time: {time_text}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display color percentages at the bottom
        y_start = frame_height - 115
        line_height = 25
        
        cv2.putText(frame, "Color Detection:", (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (color_name, percentage) in enumerate(percentages.items()):
            y_pos = y_start + (i + 1) * line_height
            color = display_colors.get(color_name, (255, 255, 255))
            
            # Format the display name
            display_name = color_name.replace('_', ' ').title()
            percentage_text = f"{display_name}: {percentage:.2f}%"
            
            cv2.putText(frame, percentage_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Moss Stress Detection with Live HSV and Percentages', frame)

        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            total_elapsed = time.time() - start_time
            end_datetime = datetime.now()
            print(f"\n{'='*80}")
            print(f"SESSION ENDED")
            print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Duration: {int(total_elapsed//3600):02d}:{int((total_elapsed%3600)//60):02d}:{int(total_elapsed%60):02d}")
            print(f"Total Frames: {frame_count}")
            print(f"Average FPS: {frame_count/total_elapsed:.2f}")
            print(f"{'='*80}")
            break
        elif key in [ord('+'), ord('=')]:
            zoom_factor = min(zoom_factor + zoom_step, 5.0)
        elif key in [ord('-'), ord('_')]:
            zoom_factor = max(zoom_factor - zoom_step, 1.0)
            if zoom_factor == 1.0:
                pan_x, pan_y = 0, 0
        elif key == ord('r'):
            zoom_factor = 1.0
            pan_x, pan_y = 0, 0
        elif key == 82:  # Up arrow
            pan_y = max(pan_y - pan_step, -100)
        elif key == 84:  # Down arrow
            pan_y = min(pan_y + pan_step, 100)
        elif key == 81:  # Left arrow
            pan_x = max(pan_x - pan_step, -100)
        elif key == 83:  # Right arrow
            pan_x = min(pan_x + pan_step, 100)

    if not use_esp32_cam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_colors_with_live_hsv(use_esp32_cam=True, esp32_url='http://10.15.75.31/cam-hi.jpg')
 
