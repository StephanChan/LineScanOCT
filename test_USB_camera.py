# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:12:11 2026

@author: shuai
"""

import cv2
import numpy as np

def nothing(x):
    pass

def test_camera_settings():
    # 1. Initialize Camera (Using MSMF as in your main code)
    # If 0 doesn't work, try 1 or 2
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    # 2. Disable Auto Exposure (0.25 is manual, 0.75 is auto for many UVC cameras)
    # Note: For some MSMF drivers, 1 is manual and 3 is auto.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    
    # 3. Set Exposure Value
    # In OpenCV/Windows, exposure is often measured in powers of 2.
    # -5 means 2^-5 = 1/32 sec (~31ms)
    # -7 means 2^-7 = 1/128 sec (~8ms)
    cap.set(cv2.CAP_PROP_EXPOSURE, 1.0)
    # Set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    # 2. Create a window with trackbars for real-time cropping adjustment
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Test', 800, 1000)

    # Trackbars for Crop (Start with your current values)
    # format: (Name, Window, Default, Max, callback)
    cv2.createTrackbar('Top', 'Camera Test', 630, 2160, nothing)
    cv2.createTrackbar('Bottom', 'Camera Test', 3670, 3840, nothing)
    cv2.createTrackbar('Left', 'Camera Test', 180, 2160, nothing)
    cv2.createTrackbar('Right', 'Camera Test', 2160, 3840, nothing)

    print("Controls:")
    print(" - Adjust trackbars to change the crop area.")
    print(" - Press 's' to save the current frame as a .tif file.")
    print(" - Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Is the camera in use by another app?")
            break

        # 3. Apply your project's Orientation (Flip + Rotate)
        # Flip horizontal (1) and then rotate 90 clockwise
        frame = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)

        # 4. Get current positions of trackbars for cropping
        top = cv2.getTrackbarPos('Top', 'Camera Test')
        bottom = cv2.getTrackbarPos('Bottom', 'Camera Test')
        left = cv2.getTrackbarPos('Left', 'Camera Test')
        right = cv2.getTrackbarPos('Right', 'Camera Test')

        # Ensure coordinates are valid (top < bottom, left < right)
        if top >= bottom: bottom = top + 1
        if left >= right: right = left + 1

        # 5. Apply Crop
        cropped_frame = frame[top:bottom, left:right]

        # 6. Display
        # We show the cropped version, and a small text overlay of the values
        display_frame = cropped_frame.copy()
        cv2.putText(display_frame, f"Crop: [{top}:{bottom}, {left}:{right}]", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{top}_{bottom}_{left}_{right}.tif"
            cv2.imwrite(filename, cropped_frame)
            print(f"Saved: {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_settings()