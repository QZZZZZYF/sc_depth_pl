import numpy as np
import matplotlib.pyplot as plt
depthmap = np.load('/media/enb/d246805e-89c3-40bf-95da-9f0d81ea7b05/home/enb/ZYF/sc_depth_pl-master/demo/480x640/outdoor/output/model_v3/depth/6.npy') #使用numpy载入npy文件
plt.imshow(depthmap) #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.savefig('depthmap.jpg') #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
plt.show() #在线显示图像


# # 保存为灰度图
# import cv2
# cv2.imwrite("depthmap.png", depthmap)

