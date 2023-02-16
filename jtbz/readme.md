## 1、yolov5s训练

先将tt100k的数据集准备好，为了防止图片大小不一，送入网络batchsize会出现问题，所以提前将数据resize到640x640，并将数据集的标注文件转换成yolo的训练格式。

标注文件中，classes种类：

![image-20230216082512755](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216082512755.png)

![image-20230216083613541](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216083613541.png)

利用python脚本将文件标注提取关键信息，然后生成yolo训练格式

![image-20230216085052914](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216085052914.png)

![image-20230216085118997](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216085118997.png)

## 2、拿到生成的新数据标签

包含45类，为常用的交通标志：

```
pl80 p6 p5 pm55 pl60 ip p11 i2r p23 pg il80 ph4 i4 pl70 pne ph4.5 p12 p3 pl5 w13 i4l pl30 p10 pn w55 p26 p13 pr40 pl20 pm30 pl40 i2 pl120 w32 ph5 il60 w57 pl100 w59 il100 p19 pm20 i5 p27 pl50
```

对应的标签通过数据集中给出的pdf文件可以查看对应的类别：例如：

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216084800542.png" alt="image-20230216084800542" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216084846119.png" alt="image-20230216084846119" style="zoom:33%;" />

都是常见的中国交通标志，然后新建yolov5s的配置文件，包括model和data

## 3、配置训练文件

model用仓库中所给的yolov5s原模型

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216085300784.png" alt="image-20230216085300784" style="zoom:33%;" />

data改成自己的训练标签和路径

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216085353381.png" alt="image-20230216085353381" style="zoom:33%;" />

## 4、训练模型，

参数使用默认参数，使用yolov5s.pt做预训练模型，调整data的路径，更改训练轮数为100个epoch（因为只是复现，所以没有训练足够，可能还需要增加轮数得到更好的效果）。

```py
python train.py --data (数据路径) --weight (预训练权重) --epoch 100 --name (保存的文件名)
```

训练完成后，可以简单的看到训练结果：

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216085900828.png" alt="image-20230216085900828" style="zoom: 33%;" />

可以通过yolov5仓库的val.py和test.py对训练出来的模型权重进行运行，因为我这里windows的环境配置没有做好，后续会改在Linux下进行。



## 5、导出onnx模型

yolov5仓库中提供了转换onnx模型的脚本，通过添加参数来导出我们需要的版本对应的onnx模型

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216090717005.png" alt="image-20230216090717005" style="zoom:33%;" />

![image-20230216090642067](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216090642067.png)

## 6、在tengine仓库对onnx模型进行优化

1、因为在yolov5-6.0版本之前yolo有一个focus算子，对于实际部署中并不有利，于是有人在yolov5的issue中提出了这个问题，将focus算子换成了卷积，在原始的模型导出中，会得到如下的模型，并不利于部署。

**Focus结构**

Focus 是 YOLOv5 新增的操作，将原始 `3*640*640` 的图像输入 Focus 结构，采用切片操作，输出 `12*320*320` 的特征图，再送入后续卷积神经网络。

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216091725443.png" alt="image-20230216091725443" style="zoom:33%;" />

2、后续优化模型直接将该slice删除，并且对算子中的sigmoid和mul算子进行了合并，形成hardswish

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216092245896.png" alt="image-20230216092245896" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216092313313.png" alt="image-20230216092313313" style="zoom:33%;" />

这个算子对NPU上来说支持比较好，推理时间大大降低。

3、对模型输出的部分进行了很大程度修改

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216092514903.png" alt="image-20230216092514903" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216092542912.png" alt="image-20230216092542912" style="zoom:33%;" />

得到了这样的一个模型，优化过后模型看起来更加精简。



## 7、在linux端对模型进行转换和量化

1、模型训练完成后下载到本地进行精度验证

![image-20230216135653302](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216135653302.png)

2、导出成onnx，同样进行精度的验证

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216135828845.png" alt="image-20230216135828845" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216135913764.png" alt="image-20230216135913764" style="zoom:33%;" />

结果：0.5mAP降低0.02   0.5:0.95mAP降低0.01

3、优化模型

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140107534.png" alt="image-20230216140107534" style="zoom:33%;" />

4、在之前编译好的tengine仓库转换模型

![image-20230216140215915](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140215915.png)

5、编写Cmakelist.txt，在Linux下进行推理，判断模型是否正确

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140644956.png" alt="image-20230216140644956" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140658420.png" alt="image-20230216140658420" style="zoom:33%;" />



<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140818327.png" alt="image-20230216140818327" style="zoom:33%;" />

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216140744392.png" alt="image-20230216140744392" style="zoom:33%;" />

6、量化UINT8

​	在提前编译好的tengine仓库里将模型转换成Uint8，一边在NPU端部署

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216141439619.png" alt="image-20230216141439619" style="zoom: 50%;" />

量化成功后得到uint8模型

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216141655460.png" alt="image-20230216141655460" style="zoom: 33%;" />

用netron可视化工具打开看到，cov层的权重已经变成了UINT8，模型参数量降低为原始模型的 1/4 倍，这个时候就可以拿到npu上进行部署了



## 8、NPU部署

在npu端侧部署遇到了许多的问题，也还有很多没有解决的问题，包括自己配置CakeList文件编译后并不生效，推理无结果，推理出现全屏框等待

1、按照同样在Linux-X86下的流程，书写cmakelist，调用timvx交叉编译的动态链接库，实现推理过程。在这之中，出现的问题在无显示结果：



2、将代码放到仓库里面进行编译，得到新的可执行文件，然后进行NPU推理，查看实验结果，发现多检测了一个P3，结果有误；但是可以通过更改置信度将这个结果抹去

![image-20230216142353130](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216142353130.png)

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216142426735.png" alt="image-20230216142426735" style="zoom:33%;" />

时间相比在x86下的cpu速度快5倍，理论推理速度是20fps，当是尝试将模型接入摄像头进行实时推理的时候发现在帧处理的时候没有做好，导致效果很差，无法实现实时推理。

3、在khadasVim3上尝试关闭NPU进行cpu推理，看到平均速度很低，相比于npu推理，慢了接近26倍，不过cpu在同样的置信度下面，并没有多检测。

![image-20230216143742407](C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216143742407.png)

<img src="C:\Users\Ruoji\AppData\Roaming\Typora\typora-user-images\image-20230216144222726.png" alt="image-20230216144222726" style="zoom:33%;" />





## 9、总结

由于之前在X86下搭建好了tengine相关的环境，导致本次实验比较快。在npu端的有关环境编译容易出比较奇怪的问题，并且由于NPU算子支持较少，为了使得模型能够更有效的在NPU上部署，需要对模型的很多算子进行优化，包括去除冗余算子（类似focus），合并算子等待操作。这一方面需要对pytorch有更深的了解，还要希望芯片厂商写更多的底层算子来支持模型的部署
