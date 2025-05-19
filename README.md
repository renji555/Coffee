# Coffee

​	1.包括一个咖啡领域训练集，综合该对话与通用对话训练集，可以使得通用大模型变得专用化

​	2.包括咖啡豆标注集

​	3.包括用于训练细粒度识别的VIT模型

​	4.使用的诸如LLaMa-Factory；Pair-ranker；YOLO系列等工具

​	5.包括经过微调后的模型

​	6.compare文件夹中的compare.py可以比较不同问题不同模型的好坏



微调后的模型可以在此下载：https://pan.baidu.com/s/1MybxSaOkV-cpO2rh8KAT3g?pwd=88dn 提取码: 88dn

使用的单个咖啡豆与多咖啡豆数据集，以及经过labelImg标注的图片信息可在此下载：
链接: https://pan.baidu.com/s/1O1TR5oAFZ6Gc-mm72ajTvg?pwd=cuhj 提取码: cuhj

咖啡豆识别的yolo参数可以参考：
链接: https://pan.baidu.com/s/1bc62B85u4-jY3k_PMXeYDw?pwd=c8gd 提取码: c8gd

## Tools

      1. transform.py: 将所有图片转换为png格式
      2. rerank.py: 将所有图片重新排序，为coffee0001、coffee0002......



# DEMO

 1. 实现摄像头获取图片并识别，具体方式见下面视频

 2. 实现咖啡对话模型，具体见下视频

 3. 参数输入界面，咱未能与控制板通信，故无法演示，只有一张示意图

    ![截屏2025-05-19 14.58.23](/Users/renhongyi/Desktop/截屏2025-05-19 14.58.23.png)

### 第三方代码引用

本项目基于以下开源代码进行调整：

1. **大模型生成效果评价**  
   - 来源：LLM-Blender (https://github.com/yuchenlin/LLM-Blender)  
   - 原作者："Jiang, Dongfu and Ren, Xiang and Lin, Bill Yuchen"  
   - 许可证：MIT License  
   - 使用位置：`compare/compare_util` 与`compare/blender.py`