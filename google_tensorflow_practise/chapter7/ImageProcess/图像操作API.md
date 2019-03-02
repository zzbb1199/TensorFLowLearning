## 图像的操作API
- 调整图片大小
  - tf.image.resize_images 直接调整大小，但有不同的调整算法
  - tf.imagae.resize_image_with_crop_or_pad 调整图片大小，带有裁剪或填充，两种操作的基点都设定在中间
- 裁剪或填充
  - 裁剪 tf.image.crop_to_bounding_box
  - 填充 tf.image.pad_to_bounding_box
- 图像翻转
- 图像色彩调整
  - 亮度
  - 对比度
  - 色相
  - 标准化
- 图片标注以及随机标注裁剪

