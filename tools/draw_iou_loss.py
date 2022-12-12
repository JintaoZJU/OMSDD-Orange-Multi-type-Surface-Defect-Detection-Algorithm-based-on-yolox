import matplotlib.pyplot as plt
import os 
import numpy as np

# cocochucheng_yolox_s_pan_baseline
# cocochucheng_yolox_s_bifpn
# cocochucheng_yolox_s_pan_se
# cocochucheng_yolox_s_pan_cbam
# cocochucheng_yolox_s_pan_cam
# cocochucheng_yolox_s_bifpn_se
# cocochucheng_yolox_s_bifpn_cbam
# cocochucheng_yolox_s_bifpn_cam
# cocochucheng_yolox_s_bifpn_se_pre_221015
# cocochucheng_yolox_s_bifpn_se_focal_ciou_pre_221015

fi_1 = open('../YOLOX_outputs/cocochucheng_yolox_s_bifpn_se_pre_221015/train_log.txt','r',encoding='utf-8')
fi_2 = open('../YOLOX_outputs/cocochucheng_yolox_s_bifpn_se_focal_ciou_pre_221015/train_log.txt','r',encoding='utf-8')

iters_num = 0 # 初始化为0
iters_num2 = 0

lines = fi_1.readlines()
lines2 = fi_2.readlines()


list_loss = []
list_loss2 = []
# total_loss iou_loss conf_loss cls_loss
for line in lines:
    if 'total_loss' in line:
        iters_num += 1  # 每得到一个损失值，+1
        #print(line)
        line = line.split('iou_loss: ')[-1].split(', l1_loss:')[0]
        list_loss.append(float(line))
    #print(line)
    #break
# print(len(list_loss))
for line in lines2:
    if 'total_loss' in line:
        iters_num2 += 1  # 每得到一个损失值，+1
        #print(line)
        line = line.split('iou_loss: ')[-1].split(', l1_loss:')[0]
        list_loss2.append(float(line))

plt.rc('font', size=10)  # 全局中英文为字体“罗马字体”
# plt.style.use('seaborn-whitegrid')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

iters_num *= 10 # 乘以10是对应txt中每10步输出一次
x = np.arange(0, iters_num, 10)  ################################ 自己估计下坐标轴x,这里10是源代码默认iter=10输出一次loss 
plt.plot(x, list_loss, label="IoU")
iters_num2 *= 10
x2 = np.arange(0, iters_num2, 10)
print(iters_num2)
plt.plot(x2, list_loss2, label="CIoU")

plt.grid(False)
plt.xlabel("Steps")
plt.ylabel("Regression_Loss")
#plt.ylim(2.0, 10.0)
plt.legend(loc="best")
plt.ylim(0,6)
#cocochucheng_yolox_s_pan_baseline
# cocochucheng_yolox_s_bifpn
# cocochucheng_yolox_s_pan_se
# cocochucheng_yolox_s_pan_cbam
# cocochucheng_yolox_s_pan_cam
# cocochucheng_yolox_s_bifpn_se
# cocochucheng_yolox_s_bifpn_cbam
# cocochucheng_yolox_s_bifpn_cam
# cocochucheng_yolox_s_bifpn_se_pre_221015
# cocochucheng_yolox_s_bifpn_se_focal_ciou_pre_221015
# plt.title("Total_loss")
#plt.annotate("Loss", (-2,10), xycoords='data',xytext=(-2,10),fontsize=15)
# plt.show()
plt.savefig('../YOLOX_outputs/exp_result_fig/cocochucheng_yolox_s_bifpn_se_pre_RegLOSS.png', bbox_inches='tight')
