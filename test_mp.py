import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
res = detector.detect(mp_image)
print('Success. Found landmarks:', len(res.pose_landmarks) > 0)
