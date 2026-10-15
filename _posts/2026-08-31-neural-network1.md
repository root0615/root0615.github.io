---
title: "인공신경망(Artificial Neural Network)과 활성화 함수(Activation Function)"
date: 2026-08-31
categories: [Pytorch]
tags: [Neural Network, Activation Function, nn.Linear, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.Softmax]
---

## 🔍 인공신경망(Artificial Neural Network)
사람의 뇌가 정보를 처리하는 방식을 본떠서 만든, 컴퓨터가 데이터에서 패턴을 스스로 찾아내도록 하는 계산 구조.<br>

![인공신경망](/assets/img/ML_DL/artificial_neural_network.png)<br>
[출처] https://blog.naver.com/qkrdnjsrl0628/222509156567<br>

### 📝 설명
우리 뇌속에는 뉴런(신경세포)이 수십억 개 있고, 이 뉴런들이 서로 연결되어 신호를 주고 받으면서 "이건 고양이다.", "이건 위험하다" 같은 판단을 내린다.<br>
인공신경망은 이걸 수학적으로 흉내낸 거라고 할 수 있다.<br>

- 뉴런 하나: 숫자 하나를 계산하는 아주 작은 계산 단위
- 뉴런들 사이의 연결: 가중치(weight), 즉 "이 신호를 얼마나 중요하게 받아들일지"를 나타내는 숫자
- 여러 뉴런이 모인층(Layer): nn.Linear 같은 층
- 여러 층이 쌓인 것: 신경망(Neural Network) 전체

### 📝 동작 흐름
1. 입력층: 데이터가 처음 들어오는 곳
2. 은닉층(hidden layer): 입력을 받아서 가중치를 곱하고, 활성화 함수(ReLU 등)를 거쳐 다음 층으로 넘김. 이 층이 여러개 쌓이면 "깊은(deep)"신경망, 즉 딥러닝이 됨.
3. 출력층: 최종 결과를 내놓음.

### 📝 핵심은 학습에 있다.
인공신경망이 특별한 이유는, 이 가중치들을 사람이 직접 정해주는 게 아니라 데이터를 보고 스스로 조정해나간다는 점.<br>

1. 처음엔 가중치가 랜덤한 값으로 시작 -> 당연히 예측 엉망
2. 모델이 낸 예측값과 실제 정답의 차이(오차, loss)를 계산
3. 이 오차를 줄이는 방향으로 가중치들을 아주 조금씩 수정(이게 역전파(backpropagation) + 경사하강법)
4. 이 과정을 데이터 전체에 대해 수천 ~ 수만 번 반복하면서 가중치가 점점 "정답을 잘 맞히는 값"으로 수렴.

## 🔍 Fully Connected NN Layer(torch.nn.Linear)
쉽게 설명하면 입력값들을 각각 다른 비중으로 섞어서 새로운 값들을 만들어내는 계산기이다.<br>

torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
- in_features: 이전 Layer에서 Node 개수 (입력되는 feature의 차원 수)
- out_feautres: 현 Layer의 Node 개수 (출력되는 feature의 차원 수)
- bias: bias 항을 사용할 것인지 정의

숫자로 직접 진행해보면
```python
import torch.nn as nn
layer = nn.Linear(in_features=2, out_features=1) 
```

이건 수식으로 `y = W1 * x1 + W2 * x2 + b` 이다.
- x1, x2는 입력값이 2개 (in_features=2)
- W1, W2는 각 입력에 곱해지는 가중치(weight) = 이 입력이 얼마나 중요한가.
- b는 bias로 입력이 0이어도 더해지는 기본값 (그래프로 치면 y절편)

### ❓ 이해를 돕기 위해 좀 더 붙이자면
만약 `out_features`가 `1`이 아닌 여러개, 예를 들어 `3`이라면 해당 수식이 출력 누련 개수만큼 반복된다.<br>

y<sub>1</sub> = W<sub>11</sub>x<sub>1</sub> + W<sub>12</sub>x<sub>2</sub> + b<sub>1</sub><br>
y<sub>2</sub> = W<sub>21</sub>x<sub>1</sub> + W<sub>22</sub>x<sub>2</sub> + b<sub>2</sub><br>
y<sub>3</sub> = W<sub>31</sub>x<sub>1</sub> + W<sub>32</sub>x<sub>2</sub> + b<sub>3</sub><br>

그리고 이걸 `행렬곱(matrix multiplication)`으로 한번에 처리한다.<br>

y = xW<sup>T</sup> + b <br>

out_features가 50개든 100개든 이 수식 하나로 처리된다.<br>
그냥 `nn.Linear(2, 50)하면 파이토치가 내부적으로 행렬 곱으로 병렬처리`한다.

### 📄 코드
```python
in_features = 128   # 입력 차원수
out_features = 64   # 출력 차원수
layer_1 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)    

print(layer_1.weight.shape)
print(layer_1.bias.shape)
"""
[코드 결과]
torch.Size([64, 128])
torch.Size([64])
"""

# layer_1의 forward pass
x = torch.rand(8, 128)
output_1 = layer_1(x)

# Weight 곱하기 X 더하기 Bias
output_2 = x @ layer_1.weight.T + layer_1.bias

# 두 값이 같은 걸 알 수 있다.
print(output_1[0, 0])
print(output_2[0, 0])
# torch.allclose(): 거의 비슷한지 확인(오차 허용)
# torch.equal(): 완전히 똑같은지 확인(가장 엄격)
print(torch.equal(output_1, output_2))
"""
[코드 결과]
tensor(-0.7334, grad_fn=<SelectBackward0>)
tensor(-0.7334, grad_fn=<SelectBackward0>)
True
"""
```

## 🔍 활성화 함수(Activation Function)
신경망의 각 층에서 나온 계산 결과에 "비선형성"을 더해주는 함수이다.<br>
그러니까, `nn.Linear에서 나온 결과를 활성화 함수에 통과`시킨다.

### ❓ 왜 이런게 필요하지?
예를 들어... `nn.Linear`는 결국 `y = Wx + b`, 즉 곱하고 더하는 것뿐이라 직선(선형) 계산이다.<br>
문제는, 이런 직선 계산을 아무리 여러 층 쌓아도 결국 하나의 직선이랑 똑같아져버린다.<br>
그런데 대부분 직선 하나로 설명이 안되는 복잡하고 구불구불한 패턴을 가지고 있다.<br>
여기에 `활성화 함수가 각 층 사이에 구부러짐을 넣어줘야, 층을 쌓을수록 점점 더 복잡한 곡선/패턴을 표현할 수 있게 된다.`<br>
흔히 사용하는 활성화 함수의 종류는 다음과 같다.<br>
Sigmoid(시그모이드), Tanh(하이퍼볼릭 탄젠트), ReLU(렐루), Softmax(소프트맥스)<br>

### 🔍 torch.nn.Sigmoid
![시그모이드](/assets/img/ML_DL/sigmoid.png)<br>
[출처] https://velog.io/@metterian/%EB%8C%80%EC%B2%B4-%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9CSigmoid-%ED%95%A8%EC%88%98%EA%B0%80-%EB%AD%94%EB%8D%B0
<br>

- 입력 x가 어떤 실수든 (음수든, 양수든) 결과값을 항상 0과 1 사이로 눌러 담아주는 함수
- x가 아주 큰 양수(+무한대)면 결과가 1에 가까워지고
- x가 아주 큰 음수(-무한대)면 결과가 0에 가까워진다.
- x가 0이면 정확히 0.5

그래서 이 함수는 확률처럼 해석하기 좋은 값(0~1 사이)을 만들 때, 특히 이진 분류(Binary Classification)에서 마지막 출력층으로 많이 쓴다.<br>
예를 들어 불량인지 아닌지 판별할 때 출력값을 0.5 기준으로 나누어 정상 / 불량 을 나눔

### 🔍 torch.nn.Tanh
![하이퍼볼릭탄젠트](/assets/img/ML_DL/hyperbolic_tansent.png)<br>
[출처] https://ko.wikipedia.org/wiki/%ED%8C%8C%EC%9D%BC:Hyperbolic_Tangent.svg
<br>

하이퍼볼릭 탄젠트는 결과 범위가 -1 ~ +1 이다.
- 아주 큰 양수를 넣으면 거의 +1
- 아주 큰 음수를 넣으면 거의 -1
- 0을 넣으면 정확히 0

시그모이드는 결과가 항상 0 이상이라서, 값들이 다 양수쪽으로 치우쳐져 다음 계산에 넘어가는 경향이 있다. <br>
하이퍼볼릭 탄젠트는 0을 중심으로 대칭이라 양수/음수가 고르게 나오니, 학습이 조금 더 안정적으로 되는 경우가 많다.<br>

### 🔍 torch.nn.ReLU
![렐루](/assets/img/ML_DL/relu.png)<br>
[출처] https://m.blog.naver.com/riverrun17/221901319498
<br>

가장 단순한 활성화 함수
- 양수면 그대로 통과시킨다.
- 음수면 무조건 0으로 만들어버린다.

Sigmoid나 Tanh는 값이 아주 크거나 작을 때 곡선이 거의 평평해져서(0이나 1에 딱 붙음), 기울기 소실 문제가 잘 생긴다.<br>
근데 ReLU는 양수 구간에서는 그냥 직선(기울기가 항상 1)이라서  이 문제가 훨씬 덜하고, 계산도 0이랑 비교해서 큰 값 고르기라 단순하고 빠르다.<br>
그래서 요즘 딥러닝에서는 가장 기본으로 많이 쓴다.<br>

### 🔍 torch.nn.Softmax
소프트맥스는 숫자 여러개를 한꺼번에 보고 확률 분포로 바꿔주는 녀석이다.<br>
즉 여러개의 답안 중에 어느 답이 맞는지 %로 계산해준다고 생각하면 된다.<br>

예를 들어 강아지/고양이/새 셋중에 하나를 분류하는 모델이 있다면 모델이 각각에 대해 점수를 낸다고 했을 때 `[2.0, 1.0, 0.1]` 이렇게 냈다고 하자. <br>
그러면 이 점수들은 크기 비교는 되지만 확률은 아니다.<br>
근데 소프트맥스는 이걸<br>

1. 점수가 큰 녀석일수록 훨씬 더 크게 부풀리고(경쟁을 심하게 만듦)
2. 전부 다 더해서 1이 되도록(=100%가 되도록) 비율을 맞춰준다

그래서 결과가 `[0.7, 0.2, 0.1]` 이렇게 나와서 강아지 70%, 고양이 20%, 새 10% 확률로 나온다.<br>
이렇게 확인해서 강아지일 확률이 제일 높으니 강아지로 판별한다.<br>
결국 서로 경쟁시켜서 이중 가장 뭐가 제일 그럴듯하냐를 정한다.(다중분류문제)<br>

### ❗실제 문제 상황
#### 📄 배경 설명
이건 제가 딥러닝을 처음 공부했을 때 했었던 실수였습니다.<br>
그때 당시 LSTM 오토인코더를 텐서플로우로 구현해보는 상황이였습니다.<br>
LSTM에 활성화 함수를 넣는 파라미터에 뭘 넣어야할지 고민했었습니다.<br>
(그냥 기본값 넣었어도 되는거였는데 하핳...)<br>
그래서 검색하다가 활성화 함수는 ReLU를 많이 사용한다기에 ReLU를 LSTM의 활성화 함수로 채택하여 학습을 시작했습니다.<br>
학습하면서 모니터링하고 있었는데 loss가 갑자기 무한대로 되더니 NaN으로 뜨면서 모델 결과가 처참해졌습니다.<br>
확인해보니 이걸 `발산했다`라고 하는것을 확인했습니다.<br>

#### ❓ 왜 발산했을까?
핵심은 `같은 값을 반복해서(=시간 스텝마다) 계속 곱해지고 더해진다`는 LSTM의 구조적 특성 때문<br>
1. LSTM은 시간축을 따라 계속 순환하는 구조이기 때문에 한 타임스텝의 출력이 다음 타임스텝의 입력으로 다시 들어가게 된다. 즉 같은 계산이 수십 ~ 수백번 반복되는 구조
2. ReLU는 위쪽 한계가 없다. 양수를 그대로 통과시켜버리니 값이 100이든 100만이든 안눌러진다.
3. 그래서 만약 어느 타임스텝이 값이 살짝 커지면, 다음 스텝에서도 그 값이 또 그대로(혹은 더 커져서)넘어가고, 그 다음 스텝에서 또 커지고... 이게 계속 누적되면서 기하급수적으로 값이 폭발해버린다. 결국 float 표현범위를 넘어서면서 inf(무한대)가 되고, inf에 어떤 연산이 걸리면 NaN이 나와버리며 발산한다.

#### ❓ 반면 Tanh는 왜 괜찮았을까??
Tanh는 아무리 큰 값이 들어와도 결과를 항상 -1 ~ +1 사이로 눌러주기 때문.<br>
그래서 시간스텝을 아무리 반복해도 값이 -1 ~ +1 범위 안에서만 움직이니까, 폭발할 수 없는 구조다.<br>
이런 이유로 LSTM은 기본 설계 자체가 원래 Tanh(그리고 게이트는 Sigmoid)를 쓰도록 만들어져있다.<br>
결론적으로 LSTM은 Tanh를 활성화 함수로 사용하는 것이 맞다.<br>
