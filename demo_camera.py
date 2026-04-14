import cv2
import torch

from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:

    ret, oriImg = cap.read()
    if not ret:
        break

    # 腿部检测
    candidate, subset = body_estimation(oriImg)
    canvas = oriImg.copy()  # 用 .copy() 替代 copy.deepcopy()，快 10 倍
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # 显示绘制骨骼输出视频帧
    cv2.imshow('openpose - legs only', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
