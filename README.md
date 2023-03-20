# Less is More: Consistent Video Depth Estimation with Masked Frames Modeling (ACM MM 2022)

[Yiran Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=p_RnaI8AAAAJ)<sup>1</sup>,
[Zhiyu Pan](https://orcid.org/0000-0001-5584-6669)<sup>1</sup>,
[Xingyi Li](https://scholar.google.com/citations?user=XDKQsvUAAAAJ&hl=zh-CN)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>1*</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

The official project of ACM MM 2022 paper "Less is More: Consistent Video Depth Estimation with Masked Frames Modeling". The code and data will be available soon.

### [Arxiv](https://arxiv.org/abs/2208.00380) | [Paper](https://github.com/RaymondWang987/FMNet/blob/main/pdf/paper.pdf) | [Supp](https://github.com/RaymondWang987/FMNet/blob/main/pdf/supp.pdf) | [Poster](https://github.com/RaymondWang987/FMNet/blob/main/pdf/MM22poster.pdf) | [Video](https://youtu.be/wvukM7WD9wE) | [视频](https://www.bilibili.com/video/BV1BD4y1z79m?spm_id_from=444.41.list.card_archive.click&vd_source=806e94b96ef6755e55a2da337c69df47)

# Abstract
Temporal consistency is the key challenge of video depth estimation. Previous works are based on additional optical flow or camera poses, which is time-consuming. By contrast, we derive consistency with less information. Since videos inherently exist with heavy temporal redundancy, a missing frame could be recovered from neighboring ones. Inspired by this, we propose the frame masking network (FMNet), a spatial-temporal transformer network predicting the depth of masked frames based on their neighboring frames. By reconstructing masked temporal features, the FMNet can learn intrinsic inter-frame correlations, which leads to consistency. Compared with prior arts, experimental results demonstrate that our approach achieves comparable spatial accuracy and higher temporal consistency without any additional information. Our work provides a new perspective on consistent video depth estimation.

![image](https://github.com/RaymondWang987/FMNet/blob/main/pdf/pipeline.PNG)

# Installation
Our code is based on `python=3.6.13` and `pytorch==1.7.1`. 

You can refer to the `environment.yml` or `requirements.txt` for installation. 

Some libraries in those files are not needed for the code.
```
conda create -n fmnet python=3.6
conda activate fmnet
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -c conda-forge
pip install numpy imageio opencv-python scipy tensorboard timm scikit-image tqdm glob h5py
```

# Demo
[Donwload](https://drive.google.com/file/d/1D2EkCEcqlImpQ15qADlfFPpdCxV_8CLt/view?usp=sharing) our checkpoint on the NYUDV2 dataset and put it in the `checkpoint` folder. 

The RGB frames are placed in `./demo/rgb`. The visualization results will be saved in `./demo/results` folder.
```
python demo.py
```

# Evaluation
[Donwload]() the 654 testing sequences of the NYUDV2 dataset and put it in the `./data/testnyu_data/` folder.

We will upload and update the link of data in this week.
```
python testfmnet_nyu.py
```


# Citation
If you find our work useful in your research, please consider to cite our paper.

```
@inproceedings{Wang2022fmnet,
  title = {Less is More: Consistent Video Depth Estimation with Masked Frames Modeling},
  author = {Yiran, Wang and Zhiyu, Pan and Xingyi, Li and Zhiguo, Cao and Ke, Xian and Jianming, Zhang},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia (MM '22)},
  year = {2022}
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
}
```
