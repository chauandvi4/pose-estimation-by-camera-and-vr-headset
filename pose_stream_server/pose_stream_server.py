"""
pose_stream_server.py
- Captures frames from a camera using OpenCV
- Runs MediaPipe Pose
- Broadcasts landmarks to connected WebSocket clients as JSON
"""

import cv2
import mediapipe as mp

import websockets
import asyncio
import json
import logging
from typing import Set

# CONFIG
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
CAM_INDEX = 0 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pose_stream_server")

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, # 0,1,2 (higher -> more accurate/slower)
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Connected websocket clients
connected_clients: Set[websockets.WebSocketServerProtocol] = set()

async def video_loop():
    cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        logger.error("Cannot open camera index %s", CAM_INDEX)
        return

    try:
        while True:
            success, image = cap.read()
            if not success:
                logger.warning("Empty frame, retrying...")
                await asyncio.sleep(0.1)
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            #Sending landmarks to connected WebSocket client
            landmarks_out = None
            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks_2d = []
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks_2d.append({
                        "id": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })

                landmarks_3d = []
                for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                    landmarks_3d.append({
                        "id": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })

                payload = json.dumps({
                    "timestamp": asyncio.get_event_loop().time(),
                    "pose_landmarks": landmarks_2d,
                    "pose_world_landmarks": landmarks_3d
                })

                # Broadcast
                if connected_clients:
                    await asyncio.wait([client.send(payload) for client in connected_clients])

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def ws_handler(ws, path):
    logger.info("Client connected: %s", ws.remote_address)
    connected_clients.add(ws)
    try:
        # Keep connection open; we don't expect messages from clients right now
        async for message in ws:
            # If client sends something (like ping/cmd) we can handle here
            logger.debug("Received from client: %s", message)
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected: %s", ws.remote_address)
    finally:
        connected_clients.remove(ws)
        
async def main():
    server = await websockets.serve(ws_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
    logger.info("WebSocket server listening on ws://%s:%s", WEBSOCKET_HOST, WEBSOCKET_PORT)
    # Run video loop forever
    await video_loop()
    server.close()
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down server.")