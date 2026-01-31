#!/usr/bin/env python3
import cv2
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="Camera index or path")
    args = parser.parse_args()

    # Handle numeric index or string path
    source = int(args.index) if args.index.isdigit() else args.index
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source {source}")
        sys.exit(1)

    window_name = f"Preview: {source} (Press 'q' to close)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    print(f"Previewing {source}... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
