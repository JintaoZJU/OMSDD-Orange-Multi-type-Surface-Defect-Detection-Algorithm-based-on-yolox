# 深度学习目标检测

## 项目介绍 (Introduction)

基于YOLOX的深度学习目标检测,用于实际生产项目中的目标检测任务，当前版本针对褚橙的表面缺陷, 模型包含模型网络实现, 以及训练、测试及转onnx格式脚本等。

### 分支功用(Branches)
写明各分支如何安排使用
- [main] 主Repo,用于存储基础功能
- [...] 其余可根据不同项目进行分支

### 开发人员(Developers)
写明项目参与人员
- 冯锦涛

## 项目使用 (Usage)

运行环境：
  - python==3.7
  - pip
  - cython 
  - pytorch::torchvision
  - pytorch::pytorch >=1.0.1
  - cudatoolkit
  - cudnn
  - pytorch::cuda100
  - matplotlib
  - git # to download COCO dataset
  - curl # to download COCO dataset
  - unzip # to download COCO dataset
  - conda-forge::bash # to download COCO dataset
  - pip:
    - opencv-python 
    - pillow <7.0 # bug PILLOW_VERSION in torchvision, must be < 7.0 until torchvision is upgraded
    - pycocotools 
    - PyQt5 # needed on KDE/Qt envs for matplotlib

项目结构：
```
| -- root
	| -- datasets
		| -- coco-format
            | -- annotations
            | -- train2017
            | -- val2017
	| -- demo
	| -- docs
    | -- tests
	| -- exps
        | -- default
            └── yolox_l/m/s/nano/s/tiny/x...
        | -- example
            | -- custom
                └── nano.py
                └── yolox_s.py
            | -- yolox_voc
    | -- tools
        └── demo.py --单图/批量 测试
        └── draw_ap/cls....py --三个损失函数、ap等曲线绘制，参数见具体函数
        └── export_onnx.py --输出onnx
        └── train.py
    | -- yolox  --模型相关关键文件夹
          | -- core 
          | -- data --数据集相关文件
            | -- datasets --数据集格式等具体修改文件都在这里
                └── coco_classes.py --coco格式类名，做了新标签到这里更新
                └── coco.py --coco数据集初始化加载
                └── voc_class.py --voc格式类名，作用同上
                └── voc.py --作用同上
            └── data_augment.py --数据增强相关文件
            └── dataloading.py --数据加载
            └── samplers.py --采样相关
            └── ...
          | -- evaluators   --评价指标相关
                └── coco_evaluator.py --coco格式评价指标 训练时验证打印表格，在train_val log中也会打印
                └── voc_eval.py --voc格式评价指标
                └── voc_evaluator.py --voc格式评价指标
          | -- exp   --模型大小等超参数设置相关
                └── base_exp.py 
                └── build.py 
                └── yolox_base.py 
          | -- layers
          | -- tools
          | -- models --模型结构相关
                └── build.py 
                └── darknet.py  --backbone
                └── losses.py   --loss
                └── network_blocks.py   --基础模块CSP,DW,SPP... 新加入SE,CBAM,CA ATTENTION定义 
                └── yolo_fpn.py
                └── yolo_head.py
                └── yolo_pafpn.py   --neck配置 Residual PAFPN实现
                └── yolox.py        --yolox网络配置      
          | -- utils
  


```

第一步,将coco格式的数据集copy至datasets路径下:
```
    注意生成数据集时id对齐,并且supercategory_id 从0开始
})
```

第二步, 配置实验文件exps:
```
    exps/examples/custom/yolox_s.py 修改depth, width, num_class, datase_path, epoch, batch_size, eval_interval...
})
```

第三步, 根据任务需求修改模型结构：
```
    yolox/models/. 对backbone更换， PAFPN中增加ATTENTION以及残差连接，损失函数更换等等
```


第四步，训练模型:
```
    python tools/train.py -f exps/example/custom/yolox_s.py -expn cocochucheng_yolox_s_bifpn_se_focal_ciou_pre_221114 --fp16 --devices 1 -c weights/yolox_s.pth 
    具体参数见train.py 自行根据任务需求修改
```

第五步, 模型推理:
```
    python tools/demo.py image -f exps/example/custom/yolox_s.py -c YOLOX_outputs/cocochucheng_yolox_s_bifpn_pre_221015/best_ckpt.pth --path datasets/test_221121/ --conf 0.3 --nms 0.65 --device gpu --save_result

```
第六步，导出onnx：
```
    python tools/export_onnx.py --output-name cocoPlum_220715.onnx -f exps/example/custom/yolox_s.py -c YOLOX_outputs/cocoPlum_220713/best_ckpt.pth
```
## 开发规范（Rules）

### git commit 风格
`[type] message`
- build: 影响构建系统或外部依赖关系的更改
- ci: 更改我们的持续集成文件和脚本
- docs: 仅文档更改
- feat: 一个新功能
- fix: 修复错误
- perf: 改进性能的代码更改
- refactor: 代码更改，既不修复错误也不添加功能
- style: 不影响代码含义的变化（空白，格式化，缺少分号等）
- test: 添加缺失测试或更正现有测试
