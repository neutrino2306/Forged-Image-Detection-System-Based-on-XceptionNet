import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# 获取图片及其路径、名称、目录
image_path = input("Please enter your image path：")
image = cv2.imread(image_path)
image_basename = os.path.basename(image_path).split('.')[0]
image_type = os.path.basename(image_path).split('.')[1]
image_dirname = os.path.dirname(image_path)

# 获得RGB版的图像，并获取图像中的全部人脸
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_detection.process(image_RGB)

# 遍历该图中所有人脸，将其一一保存
if results.detections:
    count = 1
    for detection in results.detections:
        frame = detection.location_data.relative_bounding_box
        height = image.shape[0]
        width = image.shape[1]

        # 获取按比例的左上角坐标和整个框的长宽
        x = int(frame.xmin * width)
        y = int(frame.ymin * height)
        frame_width = int(frame.width * width)
        frame_height = int(frame.height * height)

        # 计算框的中心
        x_center = x + frame_width // 2
        y_center = y + frame_height // 2

        # 宽按比例放大1.3倍，高放大1.5倍，确保额头被包括在内
        new_height = int(frame_height * 1.5)
        new_width = int(frame_width * 1.3)

        # 获取新框左上角和右下角坐标
        x1 = max(0, x_center - new_width // 2)
        y1 = max(0, y_center - new_height // 2 - int(frame_height * 0.1))
        x2 = min(width, x_center + new_width // 2)
        y2 = min(height, y_center + new_height // 2 - int(frame_height * 0.1))

        # 裁剪图像
        face = image[y1:y2, x1:x2]
        save_path = os.path.join(image_dirname, f"{image_basename}_{count}.{image_type}")
        cv2.imwrite(save_path, face)
        count += 1

else:
    print("There is no face detected in the image")

face_detection.close()

