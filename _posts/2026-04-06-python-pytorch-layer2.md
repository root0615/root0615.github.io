---
title: "[Pytorch] 레이어 정리2 Pooling Layer"
date: 2026-04-06
categories: [Pytorch]
tags: [Layer, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.MaxPool1d, nn.AvgPool1d]
---

## 💬 인삿말(잡소리)
첫번째 Linear와 Convolution 레이어에 대한 글 이후 두번째 글입니다.<br>
첫번째 글이 궁금하시다면 아래 링크로 남기겠습니다.<br>

[링크] <https://root0615.github.io/posts/python-pytorch-layer1/>

## ⚠️ 공부 목적으로 미리 작성해놓은 글입니다. 
Claude의 도움을 받아 작성했으므로 테스트 해보지 못한 코드이기에 틀릴 수 있다는 점 양해부탁드립니다.<br>
만약 글을 읽으신다면, 감안하시고 참고만 하시길 부탁드립니다.<br>
지속적으로 내용 확인하고 글을 수정할 생각입니다.<br>
그리고 제가 이해한 내용을 쉽게 풀어쓰려 노력했습니다. <br>

## 🔍 nn.MaxPool2d
지정한 윈도우 안에서 최댓값만 뽑아 공간 크기를 줄이는 레이어입니다.<br>
CNN에서 가장 널리 쓰이는 다운샘플링 방법이에요.<br>
> 커널(윈도우) 안에 있는 값 중 가장 큰 값만 선택해 출력합니다.<br>
나머지 값은 버려집니다. 학습 파라미터가 없고, 연산이 빠릅니다.<br>
출력 크기 공식: H_out = ⌊(H_in + 2×padding - dilation×(kernel-1) - 1) / stride + 1⌋<br>
비유: 사진을 4칸짜리 격자로 나눠 각 격자에서 가장 밝은 픽셀만 남기는 것.<br> 
중요한 특징(에지, 윤곽)은 살리고 불필요한 세부 정보는 압축합니다.<br>

|파라미터|설명|
|---|---|
|kernel_size|윈도우 크기.<br>2×2가 가장 일반적이며 보통 stride=2와 함께 써서 크기를 절반으로 줄입니다.|
|stride|윈도우 이동 간격.<br>기본값이 kernel_size로, 명시하지 않으면 kernel_size와 같아집니다. <br>Conv2d와 달리 기본값이 1이 아닌 점 주의!|
|padding|입력 주변에 -∞ 를 채웁니다 (0이 아닌 -∞ 임에 주의). <br>MaxPool에서는 경계 처리를 위해 씁니다.|
|dilation|윈도우 원소 간 간격. <br>넓은 영역을 희소하게 볼 때 사용합니다.|
|return_indices|True로 설정하면 최댓값의 인덱스도 함께 반환.<br>MaxUnpool2d로 역연산할 때 필요합니다.|
|ceil_mode|True이면 출력 크기 계산 시 floor 대신 ceil 사용.<br>입력이 커널로 나누어 떨어지지 않을 때 출력 크기가 1 커집니다.|

```python
import torch
import torch.nn as nn

# 가장 일반적인 패턴 — 2×2, stride=2로 크기 절반
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(8, 64, 32, 32)
out = pool(x)   # → (8, 64, 16, 16)

# stride 생략 시 자동으로 kernel_size와 같아짐 (동일 결과)
pool2 = nn.MaxPool2d(2)   # stride=2로 설정됨

# return_indices=True — MaxUnpool2d로 역연산할 때 사용
pool3 = nn.MaxPool2d(2, return_indices=True)
out, indices = pool3(x)   # indices는 역풀링에 사용

# ceil_mode=True — 홀수 크기 입력 처리
x_odd = torch.randn(1, 1, 5, 5)
pool_ceil = nn.MaxPool2d(2, stride=2, ceil_mode=True)
out_ceil = pool_ceil(x_odd)   # → (1, 1, 3, 3)  (floor면 2×2)

# dilation
pool_dil = nn.MaxPool2d(kernel_size=3, stride=1, dilation=2)
```

### ⚠️ nn.Maxpool2d 주의사항
stride 기본값이 kernel_size입니다.<br>
Conv2d처럼 기본값 1로 생각하면 오류가 생깁니다.<br>
의도적으로 overlapping pooling을 원하면 stride를 명시하세요.<br>

## 🔍 nn.AvgPool2d
윈도우 안의 모든 값을 평균 내어 출력합니다.<br>
MaxPool2d보다 부드러운 다운샘플링이 필요할 때 사용합니다.<br>

>커널 안의 모든 값을 더한 뒤 원소 수로 나눈 평균값을 출력합니다.<br> 
MaxPool과 달리 모든 값이 출력에 영향을 주므로, 역전파 시 모든 위치에 gradient가 전달됩니다.<br>
count_include_pad=True이면 padding으로 채운 0도 평균에 포함됩니다 (기본값).<br>
비유: 격자 안 모든 픽셀의 평균 밝기를 계산하는 것.<br> 
MaxPool처럼 최댓값만 취하지 않고 전체 정보를 균등하게 압축합니다.<br>

|파라미터|설명|
|---|---|
|kernel_size|평균을 계산할 윈도우 크기.|
|stride|윈도우 이동 간격.<br> MaxPool2d와 마찬가지로 기본값이 kernel_size입니다.|
|padding|입력 주변에 0을 채웁니다 (MaxPool의 -∞와 다릅니다).|
|ceil_mode|출력 크기 계산에 ceil 사용 여부.|
|count_include_pad|True이면 padding의 0도 평균 계산에 포함.<br>False이면 실제 값만으로 평균 계산.<br>경계 부분이 작아지는 현상을 막고 싶으면 False 사용.|
|divisor_override|평균 계산 시 나누는 수를 직접 지정.<br>None이면 커널 원소 수로 나눕니다.|

```python
import torch
import torch.nn as nn

# 기본 사용
pool = nn.AvgPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 64, 32, 32)
out = pool(x)   # → (8, 64, 16, 16)

# count_include_pad 차이 비교
x_small = torch.ones(1, 1, 2, 2)
p_true  = nn.AvgPool2d(kernel_size=3, padding=1, count_include_pad=True)
p_false = nn.AvgPool2d(kernel_size=3, padding=1, count_include_pad=False)
# count_include_pad=True : 9개(0 포함)로 나눔 → 모서리 값이 작게 나옴
# count_include_pad=False: 실제 값만(4개)으로 나눔 → 경계 값 왜곡 없음

# divisor_override — 나누는 수 직접 지정
pool_div = nn.AvgPool2d(kernel_size=2, divisor_override=2)
# 4칸 합을 4가 아닌 2로 나눔 → 합/2 출력
```

### 📝 MaxPool vs AvgPool
MaxPool은 가장 두드러진 특징을 강조하고, AvgPool은 전체 정보를 균등하게 압축합니다.<br>
이미지 분류엔 MaxPool이 더 자주 쓰이고, 마지막 Global Pooling 단계나 스타일 전이에는 AvgPool이 선호됩니다.

## 🔍 nn.AdaptiveAvgPool2d / nn.AdaptiveMaxPool2d
입력 크기에 관계없이 출력 크기를 고정할 수 있는 적응형 풀링 레이어입니다.<br>
kernel_size를 직접 지정하지 않아도 됩니다.<br>

>일반 AvgPool2d/MaxPool2d는 커널 크기를 고정하므로 입력 크기에 따라 출력 크기가 달라집니다.<br>
Adaptive 버전은 반대로 출력 크기를 고정하면 PyTorch가 필요한 커널 크기와 stride를 자동 계산합니다.<br>
output_size=(1,1)로 설정하면 각 채널을 스칼라 하나로 압축하는 Global Average Pooling이 됩니다.<br>
비유: 사진이 어떤 해상도든 항상 7×7 크기의 요약본으로 만들어 달라고 요청하는 것.<br>
입력이 달라져도 출력 크기는 항상 같습니다.<br>

|파라미터|설명|
|---|---|
|output_size|원하는 출력 크기.<br>정수면 정사각형 출력. <br>튜플이면 (H_out, W_out).<br> None을 쓰면 해당 축은 입력 크기 그대로 유지합니다<br> — 예: (None, 7)은 H는 그대로, W만 7로.|

```python
import torch
import torch.nn as nn

# Global Average Pooling — 어떤 크기 입력이든 (B, C, 1, 1)로 압축
gap = nn.AdaptiveAvgPool2d((1, 1))
x1 = torch.randn(8, 512, 7, 7)
x2 = torch.randn(8, 512, 14, 14)   # 다른 크기 입력도 OK
out1 = gap(x1)   # → (8, 512, 1, 1)
out2 = gap(x2)   # → (8, 512, 1, 1)  항상 같은 출력!

# 정수 하나 = 정사각형 출력
gap2 = nn.AdaptiveAvgPool2d(1)    # (1,1)과 동일
pool7 = nn.AdaptiveAvgPool2d(7)   # → (B, C, 7, 7)

# 한 축만 고정 — H는 그대로, W만 7로
pool_w7 = nn.AdaptiveAvgPool2d((None, 7))

# AdaptiveMaxPool2d — 같은 인터페이스, max 방식
amp = nn.AdaptiveMaxPool2d((1, 1))

# ResNet 마지막 분류 헤드 패턴
model_head = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),   # (B, 512, H, W) → (B, 512, 1, 1)
    nn.Flatten(),                    # → (B, 512)
    nn.Linear(512, 1000)            # → (B, 1000) ImageNet 분류
)
```

## 🔍 nn.MaxPool1d / nn.AvgPool1d
시계열·텍스트·오디오 등 1D 시퀀스 데이터에 적용하는 풀링 레이어입니다.<br>
MaxPool2d/AvgPool2d의 1차원 버전이에요.<br>

>2D 풀링과 동일한 원리지만 시간(길이) 방향으로만 동작합니다.<br>
입력 shape: (Batch, Channels, Length)<br>
출력 크기 공식: L_out = ⌊(L_in + 2×padding - dilation×(kernel-1) - 1) / stride + 1⌋<br>
비유: 음악 파형을 일정 구간으로 나눠 각 구간의 최댓값(또는 평균)만 남기는 것.<br>
시퀀스 길이를 줄이면서 핵심 패턴을 유지합니다.<br>

|파라미터|설명|
|---|---|
|kernel_size|시간 방향 윈도우 크기.|
|stride|윈도우 이동 간격.<br>역시 기본값이 kernel_size입니다.|
|padding|시퀀스 앞뒤에 패딩 추가.|
|dilation|커널 원소 간 간격.|
|return_indices|True이면 최댓값 인덱스도 반환 (MaxPool1d만 해당).|
|ceil_mode|출력 크기 계산에 ceil 사용 여부.|

```python
import torch
import torch.nn as nn

# MaxPool1d — 시계열 다운샘플링
max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
x = torch.randn(16, 32, 128)   # (B=16, C=32, L=128)
out = max_pool1d(x)             # → (16, 32, 64)

# AvgPool1d — 시계열 평균 풀링
avg_pool1d = nn.AvgPool1d(kernel_size=4, stride=2)
out2 = avg_pool1d(x)           # → (16, 32, 63)

# AdaptiveAvgPool1d — 길이 고정
gap1d = nn.AdaptiveAvgPool1d(1)    # 시퀀스 전체 평균 → 스칼라
out3 = gap1d(x)                     # → (16, 32, 1)

# 텍스트 분류에서 흔한 패턴
# Conv1d → MaxPool1d → AdaptiveAvgPool1d
text_model = nn.Sequential(
    nn.Conv1d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool1d(2),                  # 길이 절반
    nn.AdaptiveAvgPool1d(1),          # 길이 → 1
    nn.Flatten(),                     # (B, 256, 1) → (B, 256)
    nn.Linear(256, 2)                 # 이진 분류
)
```

## 🔚 마치며(잡소리)
넵, Pooling 레이어들을 알아봤습니다.<br>
글을 쓰면서 생각이 드는 건 과연 내가 이걸 또 볼까??<br>
그냥 다시 AI한테 물어보고 코딩하면 될 것 같은데?... 그럼 이렇게 쓰는 의미가 있나??<br>
이러한 의구심이 들었습니다.<br>
그런데 가끔 Claude나 GPT에게 물어보면 제가 이해할 수 없는 답변이 오는 경우가 있더라구요.<br>
그래서 이걸 좀 더 쉽게 설명해줄 수는 없나? 하고 다시 물어보면 그래도 이해되지 않았습니다.<br>
그냥 제가 지식이 너무 바닥이다보니 쉽게 설명한다 했는데 이해가 안가는거죠 ㅋㅋㅋ<br>
이렇게 글을 작성해놓고 이해한 내용으로 글을 쓰다보면 머릿속에 각인되지 않을까 싶습니다.<br>
오늘도 열심히 사시느라 고생이 많으십니다.<br>
모든일 잘 풀리길 기도해드릴게요!<br>
