import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import model
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')

# 输入图片地址，换成自己的图片地址
test_image = 'images/ski.jpg'

oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = oriImg.copy()  # 用 .copy() 替代 copy.deepcopy()

# 腿部骨骼绘制
canvas = util.draw_bodypose(canvas, candidate, subset)

# 输出结果
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.imsave('images/output.png', canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

