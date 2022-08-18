# Less is More: Consistent Video Depth Estimation with Masked Frames Modeling (ACM MM 2022)
Yiran Wang<sup>1</sup>, Zhiyu Pan<sup>1</sup>, Xingyi Li<sup>1</sup>, Zhiguo Cao<sup>1</sup>, Ke Xian<sup>1,*</sup>, Jianming Zhang<sup>2</sup>

<sup>1</sup>School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

The official project of ACM MM 2022 paper "Less is More: Consistent Video Depth Estimation with Masked Frames Modeling". The code and data will be available soon.

### [Arxiv](https://arxiv.org/abs/2208.00380) | [Paper](https://github.com/RaymondWang987/FMNet/blob/main/pdf/paper.pdf) | [Supp](https://github.com/RaymondWang987/FMNet/blob/main/pdf/supp.pdf) | [Video (Coming)]()

# Abstract
Temporal consistency is the key challenge of video depth estimation. Previous works are based on additional optical flow or camera poses, which is time-consuming. By contrast, we derive consistency with less information. Since videos inherently exist with heavy temporal redundancy, a missing frame could be recovered from neighboring ones. Inspired by this, we propose the frame masking network (FMNet), a spatial-temporal transformer network predicting the depth of masked frames based on their neighboring frames. By reconstructing masked temporal features, the FMNet can learn intrinsic inter-frame correlations, which leads to consistency. Compared with prior arts, experimental results demonstrate that our approach achieves comparable spatial accuracy and higher temporal consistency without any additional information. Our work provides a new perspective on consistent video depth estimation.

![image](https://github.com/RaymondWang987/FMNet/blob/main/pdf/pipeline.PNG)
