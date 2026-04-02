---
title: "[Pytorch] Pytorch에서 사용하는 Layer들을 쉽게 파헤쳐보자"
date: 2026-04-01
categories: [Pytorch]
tags: [Layer]
---

## 💬 인삿말(잡소리 부터 시작)
항상 딥러닝 Layer를 구성할 때 거의 Claude나 GPT의 힘을 빌리고 있습니다.<br>
그래서 그런지 Layer에 대한 기본적인 지식 개념이 상당히 부족한 것을 많이 느끼고 있어 공부하고 정리할 필요가 있다고 생각합니다.<br>
그래서 이번 글에서는 Pytorch에서 사용하는 Layer들의 사용방법을 한번 파헤치고자 글을 작성합니다.<br>
제가 모르는 Layer를 다루지 않을 수도 있는 점 참고 부탁드립니다.<br>
아! 검색하고 작성하는 데 있어 정확하지 않은 정보가 있을 수 있으니 이것도 참고 바랍니다.<br>
마지막으로 혹시 틀린 부분이 있다면, 말씀해주시면 매우 감사하겠습니다.<br>
Pytorch의 torch.nn 모듈에서 제공하는 주요 Layer들 입니다.<br>

## 🔍 nn.Linear
완전 연결층(Fully Connected Layer) -- 입력의 모든 뉴런이 출력의 모든 뉴런과 연결됩니다.
> 입력 벡터 x에 가중치 행렬 W를 곱하고 편향 b를 더하는 연산입니다.<br>
입력 차원 in_features개의 값을 받아 출력차원 out_features개의 값으로 변환합니다.<br>학습 파라미터 수는 in X out + out(bias포함)개 입니다.
```python
import torch
import torch.nn as nn
# 기본 사용법
fc = nn.Linear(784, 256) # 입력 784차원 -> 출력 256차원

x = torch.randn(32, 784)
out = fc(x)

# 파라미터 확인
print(fc.weight.shape)     # torch.Size([256, 784])
print(fc.bias.shape)       # torch.Size([256])

print(x.shape)
print(out.shape)
```
|파라미터|설명|
|---|---|
|in_features|입력 벡터의 크기. 이전 레이어의 출력 크기와 맞춰야합니다.|
|out_features|출력 벡터의 크기. 다음 레이어의 in_features와 맞춰야합니다.|
|bias|편향(b)을 학습할지 여부. 거의 항상 True. BatchNorm 레이어 바로 앞에 쓸때는 False로 줘도 됩니다.<br>(BN)이 bias 역할을 대신함|

## 🔍 Convolutional 레이어
`nn.Conv1d(in_channels, out_channels, kernel_size)`<br>
1D 데이터에 컨볼루션 적용
```python
layer = nn.Conv1d(1, 32, kernel_size=3)
```
사용 시점: 시계열 데이터, 오디오, 텍스트(NLP 일부)
|파라미터|설명|
|---|---|
|in_channels|설명|
|out_channels|
|kernel_size|
|stride|
|padding|
|dilation|
|groups|

---
`nn.Conv2d(in_channels, out_channels, kernel_size)`<br>
2D 이미지에 컨볼루션 적용 (CNN의 핵심)
```python
layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
```
사용 시점: 이미지 분류, 객체 탐지, 세그멘테이션
|파라미터|설명|
|---|---|
|in_channels|설명|
|out_channels|
|kernel_size|
|stride|
|padding|
|dilation|
|groups|

---
`nn.Conv3d(in_channels, out_channels, kernel_size)`<br>
3D 데이터에 컨볼루션 적용
```python
layer = nn.Conv3d(1, 32, kernel_size=3)
```
사용 시점: 동영상 분석, 3D 의료 영상(CT, MRI)
|파라미터|설명|
|---|---|
|in_channels|설명|
|out_channels|
|kernel_size|
|stride|
|padding|

---
`nn.ConvTranspose2d`<br>
업샘플링을 위한 역 컨볼루션
```python
layer = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
```
사용 시점: GAN 생성자, U-Net 디코더, 이미지 복원
|파라미터|설명|
|---|---|
|in_channels|설명|
|out_channels|
|kernel_size|
|stride|
|padding|
|output_padding|









