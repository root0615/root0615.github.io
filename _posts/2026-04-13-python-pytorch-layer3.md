---
title: "[Pytorch] 레이어 정리3 Normalization Layer"
date: 2026-04-13
categories: [Pytorch]
tags: [Layer, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d]
---

## 💬 인삿말(잡소리)
이번에는 Normalization 레이어들을 살펴보겠습니다.<br>
누가 이 레이어는 뭐하는 녀석이야? 라고 물어보면 답이 나와야하는데 저는 아직 멀었네요.<br>
파이토치 레이어들의 개념정리는 아래에 링크로 만들어 놓겠습니다.<br>

[링크1] Linear & Convolutional <https://root0615.github.io/posts/python-pytorch-layer1/><br>
[링크2] Pooling <https://root0615.github.io/posts/python-pytorch-layer2/><br>

## ⚠️ 공부 목적으로 미리 작성해놓은 글입니다. 
Claude의 도움을 받아 작성했으므로 테스트 해보지 못한 코드이기에 틀릴 수 있다는 점 양해부탁드립니다.<br>
만약 글을 읽으신다면, 감안하시고 참고만 하시길 부탁드립니다.<br>
지속적으로 내용 확인하고 글을 수정할 생각입니다.<br>
그리고 제가 이해한 내용을 쉽게 풀어쓰려 노력했습니다. <br>

## 🔍 nn.BatchNorm2d
미니배치 전체에 걸쳐 채널별로 정규화합니다.<br>
CNN에서 가장 널리 쓰이는 정규화 레이어입니다.<br>
BatrhNorm2d는 입력이 (N, C, H, W)일때 각 채널(C)마다 값의 크기 분포가 제각각일 수 있습니다.<br>
그러면 레이어가 학습하기 까다로워질 수 있습니다.<br>
그래서 채널별로 값을 한 번 정리해서 학습을 안정적으로 돕습니다.<br>

> 입력 (N, C, H, W)에서 채널(C) 차원을 기준으로, 같은 채널에 속하는 모든 배치·공간 위치(N, H, W)의 값들을 모아 평균과 분산을 계산해 정규화합니다.<br>
`수식: y = (x − E[x]) / √(Var[x] + ε) × γ + β`<br>
γ(scale)와 β(shift)는 학습 가능한 파라미터입니다. <br>
훈련 중엔 배치 통계를 쓰고, 추론 시엔 훈련 중 누적한 이동 평균(running_mean, running_var)을 사용합니다.<br>
비유: 반 전체 학생(배치)의 점수를 과목(채널)별로 평균 내서 상대평가하는 것.<br>
반 전체의 통계를 기준으로 각 점수를 조정합니다.<br>

|파라미터|설명|
|---|---|
|num_features|채널 수 C.<br>직전 Conv2d의 out_channels와 반드시 맞춰야 합니다.<br>정리할 서랍이 몇개인가? 라고 생각하면 됩니다.|
|eps|분모에 더하는 작은 값.<br>분산이 0에 가까울 때 수치 불안정(0으로 나누기)을 방지합니다.|
|momentum|running_mean/running_var 업데이트 속도.<br>None으로 설정하면 누적 이동 평균(단순 평균)을 사용합니다.<br>이 값은 optimizer의 momentum과는 의미가 다릅니다.|
|affine|True이면 γ(scale), β(shift)를 학습 파라미터로 갖습니다.<br>False이면 정규화만 하고 스케일/이동 없음.<br>Conv 뒤 BN에서는 True가 기본이지만, bias=False인 Conv와 쌍을 이룰 때 BN이 bias 역할도 대신합니다.|
|track_running_stats|True이면 훈련 중 running_mean/running_var를 누적합니다.<br>False이면 항상 현재 배치 통계만 사용합니다.|

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(num_features=64)

x = torch.randn(8, 64, 32, 32)
# 입력:  (B=8, C=64, H=32, W=32)
out = bn(x)
# 출력:  (8, 64, 32, 32)  ← shape은 그대로, 값만 정규화됨
# 각 채널마다 N*H*W = 8*32*32 = 8192개 값으로 평균/분산 계산

# 학습 파라미터 확인
print(bn.weight.shape)  # torch.Size([64])  ← γ (채널마다 1개)
print(bn.bias.shape)    # torch.Size([64])  ← β (채널마다 1개)

# 추론 시엔 running stats 사용
bn.eval()
out_eval = bn(x)        # running_mean, running_var로 정규화

# ─── 실전 패턴 ───────────────────────────────────
# Conv → BN → ReLU (가장 표준적인 순서)
block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    # ↑ bias=False: BN의 β가 bias 역할을 대신하므로 중복 불필요
    nn.BatchNorm2d(64),
    # (8, 3, 32, 32) → Conv → (8, 64, 32, 32) → BN → (8, 64, 32, 32)
    nn.ReLU()
    # 출력: (8, 64, 32, 32)
)

# BatchNorm1d — Linear 레이어 뒤에 사용
bn1d = nn.BatchNorm1d(num_features=256)
x1d  = torch.randn(32, 256)
# 입력:  (B=32, features=256)
out1d = bn1d(x1d)
# 출력:  (32, 256)  ← shape 그대로
```

### ⚠️ 주의 배치크기
BN은 배치 전체의 통계를 사용하므로 배치 크기가 너무 작으면(1~2) 통계가 불안정해집니다.<br>
배치가 작을 땐 GroupNorm을 사용하세요.<br>
또한 훈련/추론 모드에 따라 동작이 달라지므로 반드시 model.train() / model.eval()을 올바르게 호출해야 합니다.<br>

## 🔍 nn.LayerNorm
각 샘플 내부에서 지정한 차원들을 정규화합니다.<br>
배치 크기와 무관하게 동작해 Transformer와 RNN에서 표준으로 사용됩니다.<br>
쉽게 말하면 한 데이터 안의 숫자들을, 그 데이터 스스로의 평균과 퍼짐을 기준으로 정리하는 것입니다.<br>
그러니까 남하고 비교하는 것이 아닌 자기 자신만 보고 평균과 분산을 구해 정리한다고 생각하면 됩니다.<br>
LayerNorm은 보통 마지막 차원을 기준으로 정리합니다.<br>

> BatchNorm이 채널 방향으로 배치 전체를 보는 반면, LayerNorm은 각 샘플 하나의 지정된 차원들을 정규화합니다.<br>
배치 통계를 사용하지 않아 배치 크기=1이어도 정상 동작합니다.<br>
normalized_shape로 정규화할 마지막 차원들을 지정합니다.<br>
예: LayerNorm(512)는 마지막 차원 512에 대해 정규화합니다.<br>
비유: 한 학생(샘플)의 모든 과목 점수를 그 학생 자신의 평균으로 상대평가하는 것.<br>
다른 학생과는 무관하게 개별적으로 계산합니다.<br>

|파라미터|설명|
|---|---|
|normalized_shape|정규화할 마지막 차원의 크기.<br>int 하나면 마지막 1개 차원, 튜플이면 마지막 N개 차원을 정규화합니다.|
|eps|수치 안정성을 위한 작은 값.|
|elementwise_affine|True이면 γ, β를 학습 파라미터로 갖습니다.<br>BatchNorm의 affine과 같은 역할이지만 파라미터 이름이 다릅니다.|

> 📌 BatchNorm2d의 affine과 LayerNorm의 elemetwise_affine 비교<br>
BatchNorm2d(affine=True)이면 채널마다 학습가능한 weight와 bias을 가집니다.<br>
즉 채널수가 64면, weight 64개, bias 64개입니다.<br><br>
LayerNorm(elementwise_affine=True)이면 각 원소 위치마다 학습가능한 weight와 bias를 가집니다.<br>
- BatchNorm: “채널별로 한 번씩 조절”
- LayerNorm: “정규화하는 각 자리마다 따로 조절”

```python
import torch
import torch.nn as nn

# ─── NLP / Transformer 용도 ──────────────────────
ln = nn.LayerNorm(normalized_shape=512)

x_nlp = torch.randn(32, 100, 512)
# 입력:  (B=32, L=100, d_model=512)
out_nlp = ln(x_nlp)
# 출력:  (32, 100, 512)  ← shape 그대로
# 각 토큰(32*100=3200개)마다 512개 값으로 평균/분산 계산

# 학습 파라미터
print(ln.weight.shape)  # torch.Size([512])  ← γ
print(ln.bias.shape)    # torch.Size([512])  ← β

# ─── Transformer 블록 내부 패턴 ─────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Linear(d_model, d_model * 4)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        # x:  (B, L, 512)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)  # (B, L, 512) — Residual + LN
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)    # (B, L, 512) — Residual + LN
        return x                     # 출력: (B, L, 512)

# ─── 튜플로 여러 차원 동시 정규화 ──────────────
ln2d = nn.LayerNorm(normalized_shape=(32, 32))
x2d  = torch.randn(8, 64, 32, 32)
# 입력:  (B=8, C=64, H=32, W=32)
out2d = ln2d(x2d)
# 출력:  (8, 64, 32, 32)  ← 각 (C=64개 채널)마다 H*W=1024개 값으로 정규화
```

❓와 그래도 BatchNorm2d랑 LayerNorm이랑 어떻게 돌아가는지 모르겠다?<br>
저도 이해가 잘 가지 않아서 좀 파헤쳐봤습니다.<br>

## ✒️ BatchNorm2d / LayerNorm 비교
만약 데이터가 (N=2, C=3, H=2, W=2) 모양이라고 한다면.<br>
![데이터 예시1](/assets/img/python/Normalization1.png)<br>
이런 형태로 될 것입니다. 각 샘플 2개, 채널 3개, 높이2, 너비2 배열로 되겠습니다.

### BatchNorm2d의 계산법은?
채널별로 계산합니다.
![데이터 예시2](/assets/img/python/Normalization2.png)<br>
> 채널 0번 계산은 채널 0번만 모아서 [a, b, c, d, m, n, o, p]의 평균/분산을 계산합니다.<br>
이렇게 채널 0번, 채널 1번, 채널 2번 채널별로 계산해서 정규화를 해줍니다.<br>
그림으로 보니까 참 쉬워보이지 않습니까?? 역시 시각화를 해야...<br>
다음으로 넘어가겠습니다.<br>

### LayerNorm의 계산법은?
LayerNorm은 `normalized_shape`를 어떻게 값을 주느냐에 따라 달라집니다.<br>

📌 경우 1: nn.LayerNorm(W)<br>
이 경우는 마지막 1개 차원인 W만 봅니다.<br>
그러면 각 (N, C, H)위치에서 가로 한줄 만 보고 평균/분산 계산을 합니다.<br>
![데이터 예시3](/assets/img/python/Normalization3.png)<br>

📌 경우 2: nn.LayerNorm((W, H))<br>
이 경우는 마지막 2개 차원, 즉 (H, W)만 봅니다.<br>
그러면 각 샘플의 각 채널마다 자기 자신의 2x2만 보고 계산합니다.<br>
![데이터 예시4](/assets/img/python/Normalization4.png)<br>

📌 경우 3: nn.LayerNorm((C, W, H))<br>
이 경우는 마지막 3개 차원 전체를 봅니다.<br>
즉 한 샘플 안의 모든 채널, 모든 높이, 모든 너비를 전부 한꺼번에 모아서 평균/분산 계산을 합니다.<br>
![데이터 예시5](/assets/img/python/Normalization5.png)<br>

## 🔍 nn.GroupNorm
채널을 여러 그룹으로 나눠 그룹 내에서 정규화 합니다.<br>
배치 크기가 작아도 안정적으로 동작합니다.<br>
한 줄로 말하면, `채널을 몇 팀으로 나눠서, 팀별로 정리하는 정규화` 입니다.<br>
숫자로 예를 들면 12채널을 3개의 그룹으로 나누면 4개의 팀이 되고 팀 내부에서 정규화 시키는겁니다.<br>

> 채널 C를 num_groups개 그룹으로 나눠 각 그룹 내의 채널들과 공간(H, W)을 함께 정규화합니다.<br>
그룹 수를 조정해 BatchNorm과 InstanceNorm의 중간 특성을 조절할 수 있습니다.<br>
num_groups=1이면 LayerNorm과 동일, num_groups=C(채널 수)이면 InstanceNorm과 동일합니다.<br>
비유: 반 전체(BN)가 아니라 분단별(그룹)로 상대평가하는 것.<br>
배치 크기가 작아도 같은 분단 내 채널들끼리 정규화하므로 안정적입니다.<br>

|파라미터|설명|
|---|---|
|num_groups|채널을 나눌 그룹 수. num_channels가 반드시 num_groups로 나누어 떨어져야 합니다<br>`(num_channels % num_groups == 0)`|
|num_channels|채널 수 C. BatchNorm2d의 num_features에 해당합니다.|
|eps|수치 안정성을 위한 작은 값.|
|affine|True이면 γ, β를 학습 파라미터로 갖습니다.|

```python
import torch
import torch.nn as nn

# num_channels=64를 8개 그룹으로 → 그룹당 8채널
gn = nn.GroupNorm(num_groups=8, num_channels=64)

x = torch.randn(2, 64, 32, 32)
# 입력:  (B=2, C=64, H=32, W=32)  ← 배치 크기 2도 문제없음!
out = gn(x)
# 출력:  (2, 64, 32, 32)  ← shape 그대로
# 각 그룹(8채널)마다 H*W*8 = 32*32*8 = 8192개 값으로 정규화

# num_channels % num_groups == 0 이어야 함
gn_ok  = nn.GroupNorm(num_groups=4,  num_channels=64)  # OK: 64%4=0
# nn.GroupNorm(num_groups=6, num_channels=64)  ← Error: 64%6!=0

# ─── 특수 케이스 ─────────────────────────────────
# num_groups=1  → LayerNorm과 동일한 효과
gn_layer = nn.GroupNorm(num_groups=1, num_channels=64)

# num_groups=num_channels → InstanceNorm2d와 동일한 효과
gn_inst  = nn.GroupNorm(num_groups=64, num_channels=64)

# ─── 실전: 객체 탐지 (Mask R-CNN 스타일) ─────────
x_det = torch.randn(1, 256, 56, 56)
# 입력:  (B=1, C=256, H=56, W=56)  ← 배치=1이어도 OK
gn_det = nn.GroupNorm(num_groups=32, num_channels=256)
out_det = gn_det(x_det)
# 출력:  (1, 256, 56, 56)
```

## 🔍 nn.InstanceNorm2d
각 샘플의 각 채널 내 공간(H, W)만으로 정규화합니다.<br>
이미지 스타일 변환에 특화된 레이어입니다.<br>

>샘플 하나, 채널 하나를 독립 단위로 정규화합니다.<br>
즉 (H, W) 공간의 값들만으로 평균/분산을 계산합니다. 다른 샘플이나 다른 채널은 전혀 영향을 주지 않습니다.<br>
BatchNorm과 달리 배치 통계를 전혀 사용하지 않으므로 훈련/추론에서 동일하게 동작합니다.<br>
기본값 affine=False인 점이 BatchNorm(affine=True 기본)과 다릅니다.<br>
비유: 한 학생의 한 과목 점수를 그 과목의 문제별 점수들로만 상대평가하는 것.<br>
다른 학생, 다른 과목과 완전히 독립적입니다.

### ❓ 그럼 LayerNorm((H, W))랑 비슷한건가?
또 그건 아니랍니다.<br>
LayerNorm은 샘플 전체를 보고 계산하지만, InstanceNorm2d는 각 샘플의 채널 내 공간으로 독립적으로 계산한다고 하는데.<br>
솔직히 잘 모르겠습니다. 이해가 안가요 ㅠㅠ <br>
나중에 이해하면 내용 추가해 놓겠습니다.<br>

|파라미터|설명|
|---|---|
|num_features|채널 수 C.|
|eps|수치 안정성을 위한 작은 값.|
|momentum|running stats 업데이트 속도. <br>단, track_running_stats=True일 때만 의미가 있습니다.|
|affine|기본값이 False로 BatchNorm과 다릅니다. <br>스타일 변환에서는 보통 False로 두며, 학습 가능한 γ·β가 필요하면 True로 설정합니다.|
|track_running_stats|기본값이 False로 BatchNorm과 다릅니다. <br>기본값 False일 때는 항상 현재 입력의 인스턴스 통계를 사용합니다.|

```python
import torch
import torch.nn as nn

# 기본 사용 — affine=False, track_running_stats=False
inst_norm = nn.InstanceNorm2d(num_features=64)

x = torch.randn(4, 64, 32, 32)
# 입력:  (B=4, C=64, H=32, W=32)
out = inst_norm(x)
# 출력:  (4, 64, 32, 32)  ← shape 그대로
# 각 (샘플, 채널) 쌍마다 H*W=1024개 값으로 독립적으로 정규화
# → 총 4*64=256번의 독립 정규화 수행

# affine=True — 학습 가능한 γ, β 추가
inst_affine = nn.InstanceNorm2d(num_features=64, affine=True)
print(inst_affine.weight.shape)  # torch.Size([64])  ← γ
print(inst_affine.bias.shape)    # torch.Size([64])  ← β

# ─── 스타일 변환 (Style Transfer) 패턴 ──────────
class StyleTransferBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(channels)
        # ↑ 각 이미지가 독립적으로 정규화 → 스타일 보존에 유리
        self.relu = nn.ReLU()

    def forward(self, x):
        # x:  (B, C, H, W)
        x = self.conv(x)  # (B, C, H, W)
        x = self.norm(x)  # (B, C, H, W) ← 채널·샘플 독립 정규화
        return self.relu(x)  # (B, C, H, W)
```

## 🔚 마치며(잡소리)
지금 현재 Layer들을 공부하면서 그래도 이해가 안되는 부분이 많습니다.<br>
AI에게 물어보기에는 좀 한계가 있는 듯 합니다.<br>
뭔가 딱 이해하기 쉽게 누가 설명 좀 해줬으면 좋겠어요 ㅠ.<br>
아무튼 오늘도 화이팅입니다!!<br>

