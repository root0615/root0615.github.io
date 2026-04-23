---
title: "[Pytorch] 레이어 정리1 Linear 그리고 Convolutional"
date: 2026-04-06
categories: [Pytorch]
tags: [Layer, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d]
---

## 🔍 nn.Linear
완전 연결층(Fully Connected Layer) -- 입력의 모든 뉴런이 출력의 모든 뉴런과 연결됩니다.
> 입력 벡터 x에 가중치 행렬 W를 곱하고 편향 b를 더하는 연산입니다.<br>
수식: `y = xWᵀ + b`<br>
입력 차원 in_features개의 값을 받아 출력차원 out_features개의 값으로 변환합니다.<br>학습 파라미터 수는 in X out + out(bias포함)개 입니다.

|파라미터|설명|
|---|---|
|in_features|입력 벡터의 크기. 이전 레이어의 출력 크기와 맞춰야합니다.|
|out_features|출력 벡터의 크기. 다음 레이어의 in_features와 맞춰야합니다.|
|bias|편향(b)을 학습할지 여부. <br>거의 항상 True (default). <br>BatchNorm 레이어 바로 앞에 쓸때는 False로 줘도 됩니다.<br>(BN)이 bias 역할을 대신함|

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

print(x.shape)             # (32, 784)
print(out.shape)           # (32, 256)
```
### ❓ fc.weight의 모양이 (256, 784)가 되는 이유?
(출처) GPT<br>
출력 256개를 만들려면 출력 하나당 784개를 보는 규칙표 1개씩 필요하다는 겁니다.<br>
그러니까 쉽게 비유하자면, 정보 784개가 있고 이 정보를 보고 판단 256개를 내려야하는데..<br>
판단 1개를 하려면 784개의 정보를 각각 얼마나 중요하게 볼지 정해야합니다.<br>
판단이 256개이니 784개를 256번만큼 봐야합니다.<br>
그래서 모양이 256 * 784 배열로 만들어집니다.<br>
또 그럼 궁금한게 (784, 256)이여도 괜찮지 않느냐??<br>
이건 Pytorch가 가중치를 저장하는 방식 때문입니다...<br>
그냥 얘네들이 그런 규칙을 정한것 같습니다.<br>
또 여기서 문제점이 있습니다.<br>
데이터 배열 (32, 784)와 가중치 배열(256, 784)를 행렬곱해야 하는데 안되잖아요?<br>
그래서 가중치 배열을 전치(열과 행 뒤집음) 해줍니다 (W -> Wᵀ)<br>
그러면 행렬곱이 가능해져서 아래의 수식이 가능해집니다.<br>
수식: `y = xWᵀ + b`<br>

사용 시점:
<li>분류, 회귀 출력층</li>
<li>MLP (다층 퍼셉트론)</li>
<li>Transformer FFN</li>
<li>CNN 뒤 분류 헤드</li>
<br>

## ⚠️ nn.Linear 주의사항
Linear는 공간구조(이미지의 위치 관계)를 완전히 무시하기 때문에, 이미지 처리 중관에 끼워 넣으면 성능이 급격히 떨어집니다.<br>
이미지 중간 단계는 Conv2d를 최종 출력에만 Linear를 사용해야합니다.





## 🔍 nn.Conv2d
2D 컨볼루션 - 이미지처럼 가로/세로 공간구조가 있는 데이터에서 지역적 패턴을 추출하는 CNN의 핵심레이어입니다.

> 작은 필터(커널)를 이미지 위에서 슬라이딩하며 지역적 특징을 추출합니다. <br>
필터는 에지, 곡선, 질감 같은 패턴을 학습합니다. <br>
Linear와 달리 같은 가중치(커널)를 이미지 전체에 공유하므로 파라미터 수가 훨씬 적습니다.<br>
출력 크기 공식: H_out = (H_in + 2×padding − kernel_size) / stride + 1<br>
비유: 작은 돋보기(커널)를 이미지 위에서 조금씩 옮기며 "여기에 이런 패턴이 있네"를 감지하는 것. <br>
같은 돋보기를 전체 이미지에 재사용하므로 파라미터가 효율적입니다.

|파라미터|설명|
|---|---|
|in_channels|입력 채널 수. RGB 이미지면 3, 흑백이면1, 이전 Conv 출력이면 그 채널 수|
|out_channels|출력 채널 수 = 사용할 필터 개수. 클수록 더 많은 패턴을 학습하지만 파라미터도 증가합니다.|
|kernel_size|필터 크기. 3×3이 가장 일반적. 1×1은 채널 수 조정에, 7×7은 넓은 수용야에 사용합니다.|
|stride|필터를 몇 칸씩 이동할지. stride=2이면 출력 크기가 절반으로 줄어듭니다 (다운샘플링 효과).|
|padding|입력 주변에 0을 채우는 크기. kernel=3이면 padding=1로 입력·출력 크기를 동일하게 유지할 수 있습니다.|
|dilation|커널 원소 사이의 간격. dilation=2이면 파라미터 증가 없이 더 넓은 영역을 봅니다. 시맨틱 세그멘테이션에서 활용.|
|groups|채널을 그룹으로 나눠 독립적으로 컨볼루션. groups=in_channels으로 설정하면 Depthwise Conv가 됩니다|
|bias|편향 사용 여부. 뒤에 BatchNorm2d가 오는 경우 False 권장.|


```python
import torch
import torch.nn as nn

# 기본 Conv2d — padding=1로 입출력 크기 동일 유지
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=1)

x = torch.randn(8, 3, 32, 32)   # (B=8, C=3, H=32, W=32)
out = conv(x)                     # → (8, 64, 32, 32)

# stride=2로 다운샘플링 (MaxPool 대신 사용 가능)
conv_down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# (8, 64, 32, 32) → (8, 128, 16, 16)

# Depthwise Conv — nn.DepthwiseConv2d는 없으므로 groups로 구현
dw_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)

# Pointwise Conv (1×1) — 채널 수만 조정
pw_conv = nn.Conv2d(32, 64, kernel_size=1)

# 실전 패턴: Conv → BN → ReLU
block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),   # bias=False와 세트
    nn.ReLU()
)
"""
Tip.
Conv2d 뒤에는 거의 항상 BatchNorm2d -> Relu 순서로 붙입니다.
이때 Conv의 bias=False로 설정하는 것이 표준 패턴입니다.
"""
```

사용 시점:
<li>이미지 분류</li>
<li>객체 탐지</li>
<li>세그멘테이션</li>
<li>경량 모델</li>
<br>

### Tip. 
Conv2d 뒤에는 거의 항상 BatchNorm2d -> Relu 순서로 붙입니다.<br>
이때 Conv의 bias=False로 설정하는 것이 표준 패턴입니다.<br>


## 🔍 nn.Conv1d
1D 컨볼루션 — 시계열·오디오·텍스트처럼 시간 축(길이) 방향의 패턴을 추출합니다. Conv2d의 1차원 버전입니다.

> Conv2d가 이미지(H×W)를 처리하듯, Conv1d는 시퀀스(길이 L)를 처리합니다.<br>
커널이 시간 축 방향으로만 슬라이딩하며 지역적 패턴을 학습합니다.<br>
입력 shape: (Batch, Channels, Length)<br>
※ (Batch, Length, Channels) 순서가 아닙니다 — 순서 주의!<br>
비유: 오디오 파형 위를 작은 귀(커널)가 슬라이딩하며 "여기에 이런 소리 패턴이 있네"를 감지하는 것. <br>
Conv2d의 1차원 축소판입니다.

|파라미터|설명|
|---|---|
|in_channels|입력 채널 수. 단변량 시계열이면 1, 다변량이면 변수 수. 텍스트 임베딩이면 embedding_dim.|
|out_features|출력 채널 수 = 필터 수. 추출할 패턴 종류의 개수.|
|kernel_size|시간 방향 필터 크기. 3이면 3개 연속 시점을 한 번에 봅니다. 크면 장기 패턴, 작으면 단기 패턴.|
|stride|필터 이동 간격. stride=2이면 시퀀스 길이가 절반으로 줄어듭니다.|
|padding|시퀀스 앞뒤에 0을 채웁니다. kernel=3이면 padding=1로 길이 유지 가능.|
|dilation|커널 원소 사이 간격. 긴 시퀀스에서 파라미터 증가 없이 넓은 수용야를 확보합니다.|



```python
import torch
import torch.nn as nn

# 단변량 시계열 (채널=1)
conv1d = nn.Conv1d(in_channels=1, out_channels=32,
                   kernel_size=3, padding=1)

"""
주의 — 입력 shape 순서: PyTorch Conv1d의 입력은 반드시 (B, C, L) 순서입니다.
데이터가 (B, L, C)라면 반드시 .permute(0, 2, 1)로 바꿔야 합니다. 
특히 Embedding 레이어 뒤에서 자주 빠뜨리는 실수입니다.
"""
x = torch.randn(16, 1, 128)     # (B=16, C=1, L=128)
out = conv1d(x)                  # → (16, 32, 128)

# 다변량 시계열 (센서 6개)
x2 = torch.randn(16, 6, 128)   # (B=16, C=6, L=128)
conv_multi = nn.Conv1d(6, 64, kernel_size=5, padding=2)
out2 = conv_multi(x2)            # → (16, 64, 128)

# 텍스트 처리 — 중요: Embedding 출력은 (B, L, D)이므로 permute 필요!
embed = nn.Embedding(10000, 128)
conv_txt = nn.Conv1d(128, 64, kernel_size=3, padding=1)

tokens = torch.LongTensor([[1, 5, 3, 7, 2]])   # (1, 5)
emb_out = embed(tokens)                          # (1, 5, 128)
emb_out = emb_out.permute(0, 2, 1)              # (1, 128, 5) ← permute 필수!
txt_out = conv_txt(emb_out)                      # → (1, 64, 5)
```

사용 시점:
<li>시계열 분류, 예측</li>
<li>오디오 처리</li>
<li>NLP (n-gram 특징)</li>
<li>1D 신호 처리</li>
<br>

## 🔍 nn.Conv3d
3D 컨볼루션 — 동영상(시간+공간)이나 3D 의료 영상처럼 3개 축(Depth, Height, Width)을 가진 데이터를 처리합니다.

> Conv2d가 (H, W) 2개 축을 처리하듯, Conv3d는 (D, H, W) 3개 축을 처리합니다.<br>
동영상의 경우 D=프레임 수, H·W=공간 크기입니다.<br>
입력 shape: (Batch, Channels, Depth, Height, Width)<br>
비유: 동영상을 볼 때 "이 물체가 오른쪽으로 움직이네" 같은 시공간 패턴을 동시에 학습합니다.<br>
공간뿐 아니라 시간 방향 변화도 함께 감지합니다.

|파라미터|설명|
|---|---|
|in_channels|입력 채널 수. RGB 동영상이면 3.|
|out_features|출력 채널 수 = 필터 수.|
|kernel_size|(D_k, H_k, W_k) 형태로 각 축마다 다르게 설정 가능.<br>시간·공간 축을 다르게 쓰는 것이 흔합니다 — 예: (3,3,3) 또는 시간만 1로 줘서 (1,3,3).|
|stride|(1,2,2)이면 시간축은 그대로, 공간 크기만 다운샘플.|
|padding|각 축 패딩. kernel=(3,3,3)이면 padding=(1,1,1)로 크기 유지.|

```python
import torch
import torch.nn as nn

# 기본 Conv3d
conv3d = nn.Conv3d(in_channels=3, out_channels=32,
                   kernel_size=(3, 3, 3), padding=(1, 1, 1))

# RGB 동영상: 8프레임, 112×112
x = torch.randn(2, 3, 8, 112, 112)   # (B, C, D, H, W)
out = conv3d(x)                        # → (2, 32, 8, 112, 112)

# 공간만 다운샘플 (시간 유지)
conv3d_ds = nn.Conv3d(3, 32,
                      kernel_size=(1, 3, 3),
                      stride=(1, 2, 2),    # 시간 stride=1, 공간 stride=2
                      padding=(0, 1, 1))
out2 = conv3d_ds(x)                   # → (2, 32, 8, 56, 56)

# 3D 의료 영상 (CT 스캔)
ct_conv = nn.Conv3d(1, 32, kernel_size=3, padding=1)
ct_input = torch.randn(1, 1, 64, 64, 64)
ct_out = ct_conv(ct_input)            # → (1, 32, 64, 64, 64)
```

사용 시점:
<li>동영상 분류</li>
<li>3D 의료 영상</li>
<br>

### ⚠️ nn.Conv3d 주의사항
Conv3d는 메모리와 연산량이 매우 큽니다.<br> 
실제로는 (2+1)D Conv (공간+시간 분리) 방식이나 Transformer 기반 모델로 대체하는 경우가 많습니다.

## 🔍 nn.ConvTranspose2d
전치 컨볼루션 (Transposed Convolution) — Conv2d가 크기를 줄인다면, 이 레이어는 크기를 늘립니다. <br>
학습 가능한 파라미터로 업샘플링합니다.

> "역컨볼루션" 또는 "Fractionally Strided Convolution"이라고도 불립니다.<br>
단순 보간(bilinear)과 달리 학습 가능한 파라미터로 업샘플링하므로 더 풍부한 표현이 가능합니다.<br>
출력 크기 공식: H_out = (H_in − 1) × stride − 2×padding + kernel_size + output_padding<br>
비유: 압축된 사진을 다시 고해상도로 복원하는 것처럼, 작은 특징 맵을 학습을 통해 큰 크기로 확장합니다.<br>

|파라미터|설명|
|---|---|
|in_channels|입력 채널 수.|
|out_features|출력 채널 수.|
|kernel_size|필터 크기.<br>stride=2 업샘플링에서는 kernel_size=2 또는 4가 일반적.|
|stride|얼마나 크게 키울지 결정하는 핵심 파라미터.<br>stride=2이면 출력 크기가 2배.|
|padding|출력에서 제거할 테두리 크기.<br>Conv2d의 padding과 반대 개념.|
|output_padding|stride로 인해 생기는 크기 불일치를 보정하는 파라미터.<br>특히 홀수 크기 입력을 Conv2d로 줄였다가 복원할 때, 원본크기를 정확히 맞추기 위해서 사용합니다.<br>실제로 데이터를 추가하는 게 아닌 출력 shape 계산에만 영향을 줍니다.|

```python
import torch
import torch.nn as nn

# 2배 업샘플링 (가장 흔한 패턴)
up = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                        kernel_size=2, stride=2)
x = torch.randn(8, 64, 16, 16)
out = up(x)   # → (8, 32, 32, 32) — H, W가 2배

# U-Net 디코더 패턴 (업샘플 + skip connection)
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # skip 연결 후 채널=256
        self.bn   = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)    # skip connection 연결
        return self.relu(self.bn(self.conv(x)))

# GAN 생성자 패턴 (노이즈 → 이미지)
generator = nn.Sequential(
    nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0),  # (B,100,1,1)→(B,512,4,4)
    nn.BatchNorm2d(512), nn.ReLU(),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4→8
    nn.BatchNorm2d(256), nn.ReLU(),
    nn.ConvTranspose2d(256, 3,   kernel_size=4, stride=2, padding=1),  # 8→16
    nn.Tanh()
)
```

사용 시점:
<li>GAN 생성자</li>
<li>U-Net 디코더</li>
<li>초해상화 (SR)</li>
<li>VAE 디코더</li>
<br>

### ⚠️ nn.ConvTranspose2d 주의사항
주의 — Checkerboard 아티팩트: ConvTranspose2d는 격자 무늬 노이즈가 생길 수 있습니다.<br>
이를 피하려면 nn.Upsample(scale_factor=2, mode='bilinear') + nn.Conv2d 조합으로 대체하는 방법도 자주 사용됩니다.

## 🔚 마치며(잡소리)
다른 레이어도 정리해서 글을 올릴 예정입니다.<br>
보시고 도움이 되었으면 좋겠습니다.<br>
오늘도 화이팅입니다!!<br>

[링크2] Pooling <https://root0615.github.io/posts/python-pytorch-layer2/><br>
[링크3] Normalization <https://root0615.github.io/posts/python-pytorch-layer3/>
