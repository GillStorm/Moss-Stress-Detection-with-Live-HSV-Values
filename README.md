# Moss-Stress-Detection-with-Live-HSV-Values
 
 
Moss Stress Detection with Live HSV Values

Real-time moss health monitoring using computer vision and live HSV color detection.

This project detects and quantifies stress in moss by analyzing live video feeds (from an ESP32-CAM or local camera) and identifying color changes associated with moss health. It highlights areas of healthy green, pale green, chlorotic yellow, and necrotic brown moss using bounding boxes and calculates their percentage coverage in real-time.

Features

Live HSV detection: Continuously monitors moss color and extracts HSV values for detected regions.

Multiple input sources: Works with ESP32-CAM streaming or a local webcam.

Color stress analysis: Detects four stress categories in moss:

Healthy Green

Pale Green

Chlorotic Yellow

Necrotic Brown

Real-time overlays: Displays bounding boxes, percentages, zoom, pan, and current time on the video feed.

Digital zoom & pan: Interactive controls for zooming and panning when observing the moss closely.

Detailed statistics: Provides mean, standard deviation, and range of HSV values for each detected color in real-time.

Session summary: Prints session duration, total frames, and average FPS at the end of the session.

Controls

+ / = : Zoom in

- / _ : Zoom out

Arrow keys : Pan while zoomed

r : Reset zoom and pan

q : Quit