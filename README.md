# Less is More: Consistent Video Depth Estimation with Masked Frames Modeling (ACM MM 2022)

[Yiran Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=p_RnaI8AAAAJ)<sup>1</sup>,
[Zhiyu Pan](https://orcid.org/0000-0001-5584-6669)<sup>1</sup>,
[Xingyi Li](https://scholar.google.com/citations?user=XDKQsvUAAAAJ&hl=zh-CN)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>1*</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

The official project of ACM MM 2022 paper "Less is More: Consistent Video Depth Estimation with Masked Frames Modeling". The code and data will be available soon.

### [Arxiv](https://arxiv.org/abs/2208.00380) | [Paper](https://github.com/RaymondWang987/FMNet/blob/main/pdf/paper.pdf) | [Supp](https://github.com/RaymondWang987/FMNet/blob/main/pdf/supp.pdf)| [Poster](https://github.com/RaymondWang987/FMNet/blob/main/pdf/MM22poster.pdf) | [Video](https://youtu.be/wvukM7WD9wE) | [视频](https://www.bilibili.com/video/BV1BD4y1z79m?spm_id_from=444.41.list.card_archive.click&vd_source=806e94b96ef6755e55a2da337c69df47)

# Abstract
Temporal consistency is the key challenge of video depth estimation. Previous works are based on additional optical flow or camera poses, which is time-consuming. By contrast, we derive consistency with less information. Since videos inherently exist with heavy temporal redundancy, a missing frame could be recovered from neighboring ones. Inspired by this, we propose the frame masking network (FMNet), a spatial-temporal transformer network predicting the depth of masked frames based on their neighboring frames. By reconstructing masked temporal features, the FMNet can learn intrinsic inter-frame correlations, which leads to consistency. Compared with prior arts, experimental results demonstrate that our approach achieves comparable spatial accuracy and higher temporal consistency without any additional information. Our work provides a new perspective on consistent video depth estimation.

![image](https://github.com/RaymondWang987/FMNet/blob/main/pdf/pipeline.PNG)
