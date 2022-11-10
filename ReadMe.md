该repository的目的：

- 复现了相关 轻量级模型，目前已实现MobileViT模型， 以及分布式训练的范式
- 快速图像解码和高效数据预加载，加速训练速度
- 分布式训练目前实现了pytorch的分布式训练



To Do： 

 - 复现 MobileViT V2模型，MobileViT V3模型 以及 NextViT模型等
 - 后续通过加入Nvidia的Dali框架加载图像训练数据，加速训练
 - 实现基于hugging face 的 accelerate的分布式训练