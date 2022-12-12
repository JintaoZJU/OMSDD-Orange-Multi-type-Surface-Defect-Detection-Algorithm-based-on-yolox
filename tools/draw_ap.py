import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os 
import numpy as np

# fi_1 = open('../YOLOX_outputs/cocochucheng_yolox_s_pan_baseline/train_log.txt','r') #,encoding='utf-8')  # ################################ 1、修改路径
fi_1 = open('../YOLOX_outputs/cocochucheng_yolox_s_bifpn/train_log.txt','r')
epoch_nums = 44  # ###################### 2、修改对应自己的训练总epoch数（对应下面x坐标）

lines = fi_1.readlines()

list_AP = []
list_AP50 = []
list_AP75 = []
list_APM = []
list_APL = []
list_AR100 = []


for line in lines:
    if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
        # print(line[-6:])
        list_AP.append(float(line[-6:]))
    elif 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
        # print(line[-6:])
        list_AP50.append(float(line[-6:]))
    elif 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]' in line:
        # print(line[-6:])
        list_AP75.append(float(line[-6:]))
    elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]' in line:
        # print(line[-6:])
        list_APM.append(float(line[-6:]))
    elif 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in line:
        # print(line[-6:])
        list_APL.append(float(line[-6:]))
    elif 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
        # print(line[-6:])
        list_AR100.append(float(line[-6:]))

# print(list_AP)
# print(list_AP50)
# print(list_AP75)
# print(list_APM)
# print(list_APL)
# print(list_AR100)
plt.rc('font',  size=6)  # 全局中英文为字体“罗马字体”
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'



x = np.append(np.arange(0, 285, 10), np.arange(285, 300, 1)) ######################################### 对应自己epoch
plt.plot(x, list_AP, label="AP(IOU=0.5:0.95)")
plt.plot(x, list_AP50, label="IOU=0.5")
plt.plot(x, list_AP75, label="AP(IOU=0.75)")
plt.plot(x, list_APM, label="medium")
plt.plot(x, list_APL, label="AP(Large)")
plt.plot(x, list_AR100, label="AR(maxDets=100)")

plt.xlabel("Epoch")
plt.xlim(0,300)
plt.xticks(x) #################################################### 同上 



x_major_locator = MultipleLocator(10)  # 把x轴的刻度间隔设置为10，并存在变量里 ############################### 设置坐标轴间隔
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.grid(False)
plt.legend(loc="best")
plt.title("cocochucheng_yolox_s_bifpn_ap")
# plt.show()
plt.savefig('../YOLOX_outputs/cocochucheng_yolox_s_bifpn/cocochucheng_yolox_s_bifpn_AP.png', bbox_inches='tight')


