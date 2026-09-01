---
title: "파이토치의 모델 구조"
date: 2026-08-31
categories: [Pytorch]
tags: [FCN, nn.Sequential, nn.Module]
---

이번에도 nn.Linear를 설명하면서 글을 쓰기 시작합니다.<br>
모든 글이 제가 그냥 정리하면서 공부하는 글이어서 누가 볼지는 모르겠지만<br>
제가 나중에 개념을 잊어버려서 다시 찾아봤을 때 바로 이해할 수 있도록 쓰려고 노력하고 있습니다.<br>

## 🔍 Fully Connected Network (FCN, 완전연결신경망)
정의하자면, 이전 층의 모든 뉴런이 다음 층의 모든 뉴런과 하나도 빠짐없이 연결되어 있는 가장 기본적인 신경망 구조입니다. nn.Linear(파이토치) / Dense(텐서플로우)가 이 연결 방식 자체를 구현한 층입니다.<br>
기본적인 구조는 `nn.Linear`와 `활성화 함수`로 이루어져 있으며 추가적으로 정규화를 하기 위한 `nn.BatchNorm1d`이나 과적합 방지를 위한 `nn.Dropout`이 섞여있어도 완전연결신경망이라고 부릅니다.<br>
그리고 완전연결신경망을 다른이름으로 MLP(Multi-Layer Perceptron, 다중 퍼셉트론)이라고도 부릅니다.<br>

### 💬 쉽게 설명?
비유로 하자면 회의실에 2그룹이 있다고 한다면 A그룹(입력층) 사람들이 B그룹(출력층) 사람들한테 각자
자기 의견을 전달하는데, A그룹의 모든 사람한테 개별적으로 말을 건넨다라고 생각하면 됩니다.<br>
A그룹이 5명이고 B그룹이 4명이면 총 5 * 4 = 20개의 연결선(대화)이 생긴다는 거죠.<br>
이 연결선 하나하나가 각각 다른 가중치(weight)를 가지고 있어서, 어떤 연결은 더 중요하게(가중치 크게),
어떤 연결은 덜 중요하게(가중치 작게) 반영합니다.

### 🔍 장단점 및 실무 위치
- 장점: 구조 단순, 어떤 형태의 데이터든 (일단 1차원으로 펴기만 하면) 넣을 수 있어서 범용적
- 단점: 모든 걸 다 연결하다보니 파라미터 수가 폭발적으로 늘어남, 그래서 비효율적이고 과적합도 잘 발생
- 실무에서는 CNN / LTSM으로 특징을 먼저 추출한 뒤, 맨 마지막에 FC층을 붙여서 최종 결과(분류/회귀)로 종합하는 역할로 많이 씀.

### 💻 코드
```python
import torch

x = torch.rand(8, 64)
print(f"Layer 통과전 원본: {x.shape}")

# First Layer
# feature vector를 64 -> 32 변환
# FC Layer + Tanh으로 구성
layer1 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
act_fn1 = torch.nn.Tanh()

# Second Layer
# feature vector를 32 -> 16 변환
# FC Layer + Tanh으로 구성
layer2 = torch.nn.Linear(in_features=32, out_features=16, bias=True)
act_fn2 = torch.nn.Tanh()

# Third Layer
# feature vector를 16 -> 1 변환
# FC Layer + Sigmoid으로 구성
layer3 = torch.nn.Linear(in_features=16, out_features=1, bias=True)
act_fn3 = torch.nn.Sigmoid()

# Layer 층 계산 후 활성화 함수 계산
out1 = act_fn1(layer1(x))
print(f"First Layer 통과: {out1.shape}")

out2 = act_fn2(layer2(out1))
print(f"Second Layer 통과: {out2.shape}")

out3 = act_fn3(layer3(out2))
print(f"Third Layer 통과: {out3.shape}")
"""
[코드 결과]

Layer 통과전 원본: torch.Size([8, 64])
First Layer 통과: torch.Size([8, 32])
Second Layer 통과: torch.Size([8, 16])
Third Layer 통과: torch.Size([8, 1])
"""
```

## 🔍 nn.Sequential
한줄로 설명하면 파이토치에서 `여러 층(Layer)을 순서대로 쭉 이어붙여서 하나의 모델로 묶어주는 상자 같은 역할`을 한다.

### 💬 쉽게 설명?
공장의 컨베이어 벨트를 생각하면 된다.<br>
원재료(입력데이터)가 벨트를 타고 들어가서, 1번 작업대를 거치고 --> 2번 작업대 --> 3번 작업대 --> 완제품(출력)이 나온다.<br>
nn.Sequential은 이 작업대들을 순서대로 세팅해놓은 벨트 자체라고 보면 된다.<br>
벨트에 물건(데이터)을 넣기만 하면, 순서대로 자동으로 하나씩 거쳐서 나온다.<br>

### 💻 코드
```python
import torch

x = torch.rand(8, 64)

# 기존 코드 모양 ===================
layer1 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
act_fn1 = torch.nn.Tanh()
layer2 = torch.nn.Linear(in_features=32, out_features=16, bias=True)
act_fn2 = torch.nn.Tanh()
layer3 = torch.nn.Linear(in_features=16, out_features=1, bias=True)
act_fn3 = torch.nn.Sigmoid()
out1 = act_fn1(layer1(x))
out2 = act_fn2(layer2(out1))
out3 = act_fn3(layer3(out2))
print(out3.shape)

# nn.Sequential 사용 ==================
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=64,
                    out_features=32,
                    bias=True),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features=32,
                    out_features=16,
                    bias=True),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features=16,
                    out_features=1,
                    bias=True),
    torch.nn.Sigmoid()
)

# 위처럼 Layer를 하나로 묶어서 정의 하고 아래와 같이 입력하면 된다.
out = model(x)
print(out.shape)
"""
[코드 결과]

torch.Size([8, 1])
torch.Size([8, 1])
"""
# 동일한 형태의 데이터가 나온 것을 확인할 수 있다.
```

### 💬 언제 쓰고 언제 안쓰는게 좋을까??
- 데이터가 층을 하나씩 순서대로만 통과하면 되는 단순한 구조일 경우 사용
- 층 중간에 복잡한 계산이나 `skip connection` 처럼 입력을 몇층 건너뛰어서 나중에 더해줘야하는 구조일 경우 사용하지 않는다.
- 그래서 가장 효울적인 방법은 구조가 한줄로 쭉 이어지는 부분(FC 층 여러개 쌓기)은 Sequential로 묶고 복잡한 부분은 직접 짜는 것이 좋다.

#### ❓skip connection 설명
한 층을 건너뛰어서, 입력을 몇층 뒤의 출력에 그대로 더해주는 지름길<br>
왜 필요할까?? 층을 아주 깊게 쌓으면(수십 ~ 수백 층), 학습 신호(gradient)가 뒤에서 앞으로 전달되는 과정에서 점점 약해지거나 사라지는 문제(기울기 소실)가 발생. <br>
그래서 앞쪽 층들은 거의 학습이 안되는 상황 발생.<br>
그렇기에 원본 입력을 몇층 건너뛰어 뒤쪽 출력에 그대로 더해줌으로써, 깊은 네트워크에서도 학습 신호가 잘 전달되게 해주는 지름길이다.<br>

y = F(x) + x<br>

하지만 이상탐지에서는 skip connection이 너무 강하면 모델이 압축된 특징을 학습하는 대신 그냥 원본을 거의 그대로 배껴서 복원할 가능성 있음.<br>
그렇게되면 데이터를 구분하는 능력이 오히려 약해질 수 있다.<br>
그래서 이상탐지용 오토인코더에서는 아예 안쓰거나, 일부 층에만 제한적으로 사용<br>

## 🔍 nn.Module
nn.Module은 파이토치에서 모든 신경망 부품(층이든, 전체 모델이든)의 기본 설계도(부모 클래스)이다.<br>
nn.Linear, nn.LSTM, nn.Sequential, 그리고 직접 만든 커스텀 모델까지 전부 이 nn.Module을 상속받아 만들어진다.<br>

### 📄 특징 설명
- nn.Module을 상속받은 모든 클래스는 신경망 모듈이 된다.
- 신경망 모듈은 GPU로 이동할 수 있고, 내보내기(exporting), 불러오기(loading)가 가능하며, 복합적인 구조를 가질 수 있다.
- nn.Module은 또한 Parameter를 포함하고 있는데, 이는 학습 중에 최적화되는 가중치와 편향이다.

비유하자면 레고블록과 같다. 레고블록이 되기위한 공통 규격과도 같아서 
- 다른 블록이랑 쉽게 결합 가능(여러 층을 쌓아서 하나의 모델로 만듦)
- 블록 안에 뭐가 들어있는지(파라미터가 몇 개 인지)자동으로 관리
- GPU로 옮기거나, 저장하거나, 불러오는게 다 똑같은 방식으로 됨

### 📄 지켜야할 규칙
직접 nn.Module을 사용하려면 기본적으로 지켜야할 2가지가 있는데 그건 다음 두 Method을 Override하여 정의해주어야 한다.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()   # 1. 부모(nn.Module)한테 "나 초기화 좀 해줘" 라고 알려줌
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):    # 2. "데이터가 들어오면 어떤 순서로 계산할지" 정의
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

1. __init__: 이 모델이 어떤 부품(층)들로 이루어져 있는지 "선언"만 해두는 곳
2. forward: 실제로 데이터가 들어왔을 때 그 부품들을 어떤순서로, 어떻게 쓸지 정의하는 곳

#### ❓ `super().__init__()`을 꼭 써야하는 이유
이걸 빼먹으면 에러가 난다. <br>
왜냐하면 nn.Module이 내부적으로 "이 모델 안에 어떤 학습 가능한 파라미터(가중치)들이 있는지"<br>
자동으로 추적하는 시스템을 갖고 있는데, 이 초기화 코드가 먼저 실행되어야 그 추적 시스템이 세팅된다.<br>
이게 없으면 self.fc1 = nn.Linear(10, 20)을 써도 파이토치가 이걸 학습해야할 부품으로 인식을 못한다.<br>

📌 nn.Module이 자동으로 해주는 것들(핵심) <br>

```python
model = MyModel()

model.parameters()      # 모델 안의 모든 가중치를 자동으로 다 찾아서 알려줌
model.to('cuda')        # 모델 전체를 GPU로 한번에 옮겨줌
model.state_dict()      # 모델의 모든 가중치를 저장 가능한 형태로 뽑아줌
model.eval()            # 평가 모드로 전환 (Dropout, BatchNorm 등 동작 방식 자동 조정)
model.train()           # 학습 모드로 전환
```

### 💻 코드

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        # super().__init__()와 완전히 같은 동작 수행.
        super(Model, self).__init__()
        """
        super(Model, self).__init__(): python2 시절 부터 쓰던 정식 문법
        >> "Model 클래스의 부모를, self 기분으로 초기화 해줘" 라고 클래스 이름과 인스턴스를 직접 명시하는 방식

        super().__init__(): python3부터 추가된 축약형
        >> 파이썬이 알아서 현재 클래스와 self를 자동으로 채워줌. 위와 똑같은 동작을 훨씬 짧게 해줌.

        결론적으론 super().__init__() 이걸 쓰는게 맞고 사실상 표준(정석)이다.
        """
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=64,
                            out_features=32,
                            bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=32,
                            out_features=16,
                            bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=16,
                            out_features=1,
                            bias=True),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

model = Model()
x = torch.rand(32, 64)
out = model(x)

print(out.shape)
print(model)        # 모델 레이어 구조 확인 가능
"""
[코드 결과]

torch.Size([32, 1])
Model(
  (layers): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): Tanh()
    (2): Linear(in_features=32, out_features=16, bias=True)
    (3): Tanh()
    (4): Linear(in_features=16, out_features=1, bias=True)
    (5): Sigmoid()
  )
)
"""
```

### 🔍 Model Summary
파이토치 모델의 구조를 층별로 한눈에 보여주는 라이브러리<br>
torchsummary을 통해서 모델의 구조, 각 Layer의 parameter개수, 각 Layer 출력값의 차원 수 등을 파악할 수 있다.<br>
```python
from torchsummary import summary

"""
input_size에는 패치크기를 빼고 입력 하나의 shape만 적어줘야한다.
실제로 넣는 데이터가 (32, 64)이므로 여기엔 (64,)만 적으면 된다.
torchsummary가 내부적으로 배치 크기 1짜리 더미 데이터를 자동으로 만들어서 모델에 통과시키는 방식으로 동작한다.

device의 경우 summary는 내부적으로 기본값이 GPU("cuda")로 되어있기 때문에
모델이 현재 cpu에 있다면 device="cpu"로 맞춰줘야한다.
"""
print(summary(model, input_size=(64,), device="cpu"))
"""
[코드 결과]

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 32]           2,080
              Tanh-2                   [-1, 32]               0
            Linear-3                   [-1, 16]             528
              Tanh-4                   [-1, 16]               0
            Linear-5                    [-1, 1]              17
           Sigmoid-6                    [-1, 1]               0
================================================================
Total params: 2,625
Trainable params: 2,625
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.01
Estimated Total Size (MB): 0.01
----------------------------------------------------------------
None
"""
```