# Coffee

​	1.包括一个咖啡领域训练集，综合该对话与通用对话训练集，可以使得通用大模型变得专用化

​	2.包括咖啡豆标注集

​	3.包括用于训练细粒度识别的VIT模型

​	4.使用的诸如LLaMa-Factory；Mix-instructV2；YOLO系列工具

​	5.包括经过微调后的模型

​	6.compare文件夹中的compare.py可以比较不同问题不同模型的好坏



微调后的模型可以在此下载：https://pan.baidu.com/s/1MybxSaOkV-cpO2rh8KAT3g?pwd=88dn 提取码: 88dn

### 第三方代码引用
本项目基于以下开源代码进行调整：

1. **大模型生成效果评价**  
   - 来源：LLM-Blender (https://github.com/yuchenlin/LLM-Blender)  
   - 原作者："Jiang, Dongfu and Ren, Xiang and Lin, Bill Yuchen"  
   - 许可证：MIT License  
   - 使用位置：`compare/compare_util` 与`compare/blender.py`