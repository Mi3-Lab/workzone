import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import sys

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(f"Connected to RealSense Device: {device_product_line}")

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        
        if not found_rgb:
            print("Error: The device does not have an RGB Camera sensor.")
            return

        # Configure the color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("\nPossible fixes:")
        print("1. Ensure the RealSense camera is connected.")
        print("2. Install pyrealsense2: pip install pyrealsense2")
        print("3. On Jetson, you might need to build from source or use a specific wheel.")
        return

    print("\nControls:")
    print(" - Press [SPACE] to start 5s countdown and save picture.")
    print(" - Press [Q] or [ESC] to quit.")

    countdown_start_time = None
    counting_down = False
    
    # Create directory for images if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "captured_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()

            if counting_down:
                elapsed = time.time() - countdown_start_time
                remaining = 5 - int(elapsed)
                
                if remaining > 0:
                    # Draw countdown on the image
                    text = f"Capturing in {remaining}s..."
                    font = cv2.FONT_HERSHEY_DUPLEX
                    scale = 2
                    color = (0, 0, 255) # Red in BGR
                    thickness = 4
                    
                    # Center the text
                    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
                    x = (display_image.shape[1] - text_size[0]) // 2
                    y = (display_image.shape[0] + text_size[1]) // 2
                    
                    # Add a simple shadow for readability
                    cv2.putText(display_image, text, (x+2, y+2), font, scale, (0, 0, 0), thickness)
                    cv2.putText(display_image, text, (x, y), font, scale, color, thickness)
                else:
                    # Save the high-quality original frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"realsense_capture_{timestamp}.jpg")
                    cv2.imwrite(filename, color_image)
                    print(f"\n[SUCCESS] Saved image: {filename}")
                    
                    # Flash effect
                    display_image = np.full_like(display_image, 255)
                    cv2.imshow('RealSense Preview', display_image)
                    cv2.waitKey(100)
                    
                    counting_down = False

            # Show the preview
            cv2.imshow('RealSense Preview', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or Esc
                break
            elif key == ord(' ') and not counting_down:
                print("Space pressed! Starting 5-second countdown...")
                countdown_start_time = time.time()
                counting_down = True

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped. Goodbye!")

if __name__ == "__main__":
    main()
