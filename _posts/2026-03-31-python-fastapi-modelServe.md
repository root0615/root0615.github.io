---
title: "[Python] 간단하게 FastAPI로 Model Serving 해보기"
date: 2026-03-31
categories: [Python]
tags: [FastAPI, Model Serving, CNNAutoencoder, Uvicorn, Pyinstaller]
---

## 💬 인삿말(잡소리 부터 시작)
안녕하세요. <br>
제가 하는 일은 설비에서 수집한 신호데이터를 AI 모델에 학습하여 정상/불량을 판정하는 일을 담당하고 있습니다. <br>
그래서 학습한 모델을 설비에 배포해서 사용해야하는데 저희 프로그램 C#으로 만들었어요. <br>
그래서 python에서 C#으로 모델 판정 결과를 전달하는 방법을 쓰고 있는데 제가 현재 다니는 직장에서는 독자적인? 방법을 사용합니다.<br>
그래서 대중적으로 효율적이고 간단하게 할 수 있는 방법을 찾았습니다. <br>
물론 목표에서는 벗어나면 안되겠지요. <br>
그래서 알아본 것은  Model Serving이라는 개념을 알았고 FastAPI로 하는 방법을 공부했습니다. <br>
네, 저는 그렇게 전문가가 아니라 지금 알아버렸습니다. 흑... 아무튼 이제 본론으로 돌아가겠습니다.

## 🔍 Model Serving 그리고 FastAPI
> (제미나이 왈) <br>
`모델 서빙`은 학습된 머신러닝을 실제 서비스 환경(운영 환경)에 배포하여, 사용자나 어플리케이션의 요청에 따라 실시간으로 예측 결과(추론)를 제공하는 기술 및 과정입니다. API 형태로 모델을 호스팅하여 전/후처리 과정을 포함한 추론 결과를 안정적으로 반환하는 것이 핵심입니다.<br>
`FastAPI`는 파이썬 기반으로 현대적이고 빠른(고성능) API를 구축하기 위한 웹 프레임워크입니다.

그래서 FastAPI로 환경을 구축하고 단순 모델서빙을 진행해보았습니다. <br>
코드를 보기 전에 먼저 구조를 먼저 소개하고 가는게 이해하기 편할 것 같아 이미지를 통해 말씀드리겠습니다.

## 📝 작업구조
`이게 정답은 아니니까 알아서 구조는 짜셔야합니다!`

![FastAPI 작업 구조](/assets/img/python/FastAPI_modelServing_architecture.png)

|구조|설명| 
|---|---|
|`main.py`|FastAPI 앱의 핵심 파일입니다. 서버 시작 시 모델을 로드하고 요청을 받아 처리합니다.| 
| `inference.py` |모델 로드와 추론을 담당합니다.| 
| `model.py` |학습한 모델 구조를 정의 합니다.| 
| `cnn_ae_fin.pt` |학습된 모델의 정보를 담은 파일입니다. 저는 CNN AutoEncoder를 사용했습니다.| 
| `logger.py` |파이썬 내장 로그 모듈, 콘솔과 파일 두 곳에 기록하게 설정| 
| `logs/` |날짜별 로그 파일이 쌓이는 폴더입니다.| 
| `uvicorn` |FastAPI를 실행시켜주는 웹 서버입니다. 실제로 HTTP 요청을 받아 FastAPI에게 전달합니다.| 
| `Client` |CSV 파일을 읽어 JSON으로 변환 후 FastAPI에 요청합니다.|

> 저장된 모델 파일(모델.pt)을 클라이언트에게 전달하기 위해 학습한 모델 구조를 model.py에서 정의. <br>이것을 inferenc.py에서 모델 로드와 추론 메서드를 생성. main.py에서 요청을 받아 inference.py를 참조하여 모델 추론 결과 생성 후 Client에게 전달. <br>이렇게 구현했습니다.

## 📝 폴더 구조
```
C:\fastapi\
├── main.py
├── logger.py
└── models\
    └── CNNAutoencoder\
        ├── model.py
        ├── inference.py
        └── cnn_ae_fin.pt
```
---
다시 한번 말씀드리지만 이게 정답이 아니라는 점 강조 한번 더 하겠습니다.<br> 그만하라고요? 알겠습니다.<br>
아 그리고 logger의 경우 제가 저번에 썼던 글 그대로 사용한거니 참고하시면 되겠습니다.<br>
[참조] <https://root0615.github.io/posts/python-logger><br>

## 📄 코드
가장 중요한 역할을 하는 main.py 파일만 확인해보면 될것 같습니다.
```python
# 파일명 : main.py
import time
# 서버 시작/종료 시 실행할 코드를 정의하기 위한 데코레이터
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
# 요청/응답 스키마 정의할 때 사용
from pydantic import BaseModel

from logger import logger

# inference.py의 모델로드 및 추론을 담당하는 AnomalyDetector 클래스를 불러옵니다.
from models.CNNAutoencoder.inference import AnomalyDetector

# 클라이언트가 보내는 JSON 형태를 정의 및 검증
class PredictRequest(BaseModel):
    data: list[list[float]]

# 서버가 반환하는 JSON 형태 정의
class PredictResponse(BaseModel):
    recon_error: float
    threshold: float
    is_anomaly: bool
    status: str



"""
# 서버 시작 시 모델 가져오기 (1번만 실행)
# @asynccontextmanager 데코 덕분에 
yield 기준으로 시작 / 종료가 나뉘어짐
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        """
        # app.state는 FastAPI가 제공하는 전역 저장소로 모든 요청에서 접근 가능합니다.
        'app.state.원하는변수명 = 저장할 객체' 이렇게 저장합니다.
        위와 같이 저장하면 서버가 살아있는 동안 계속 유지되기에 해당 방법으로 사용했습니다.
        여기에 원하는 모델을 불러와주면 되겠습니다. 
        저는 CNNAutoencoder 모델을 로드하는 AnomalyDetector 클래스를 미리 정의하여 불러왔습니다.
        """
        app.state.detector = AnomalyDetector()
        logger.info("model load complete")

    # 예외처리
    except FileNotFoundError as e:
        logger.error(f"model file not found: {e}")
        app.state.detector = None
    except KeyError as e:
        logger.error(f"checkpoint Key Error: {e}")
        app.state.detector = None
    except RuntimeError as e:
        logger.error(f"model load failed: {e}")
        app.state.detector = None
    yield
    logger.info("FastAPI server shutdown")

# FastAPI 앱을 생성하면서 lifespan을 등록합니다.
# lifespan에 등록해야 서버 시작/종료 시 위에서 정의한 코드가 실행됩니다.
app = FastAPI(
    title="FastAPI server",
    lifespan=lifespan,
)

# health check(서버가 정상 동작 중인지 확인하는 엔드포인트) 모니터링 용
@app.get("/health")
def health():
    if app.state.detector is None:
        logger.error(f"Model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )
    return {"status": "ok"}

# 판정 포인트, response_model을 지정해주어 해당 스키마(PredictResponse)로 JSON형태로 반환합니다.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    # 모델 상태 확인
    if app.state.detector is None:
        logger.error(f"Model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )
    
    # 여기서부터는 데이터 가져와서 전처리 시작
    data = np.array(req.data, dtype=np.float32)

    # 데이터 쉐이프 검증
    if not data.shape[1] == 8:
        logger.warning(f"data shape is incorrect: {data.shape}")
        raise HTTPException(
            status_code=422,
            detail=f"데이터 채널이 8이어야 합니다. 현재: {data.shape[1]}"
        )

    # 추론 및 처리 시간 측정
    start = time.time()
    # 판정
    result = app.state.detector.predict(data)
    elapsed = time.time() - start

    logger.info(
        f"recon_error={result['recon_error']:.6f} | "
        f"threshold={result['threshold']:.6f} | "
        f"is_anomaly={result['is_anomaly']} | "
        f"status={result['status']} | "
        f"inference time={elapsed*1000:.1f}ms"
    )

    return result
```
여기서 일단 중요한것은 모델을 한번만 불러오는 것입니다. 
> `@asynccontextmanager` 데코레이터는 yield를 기준으로 앞은 시작 시, 뒤는 종료 시 실행되는 구조를 만들어줍니다. <br>모델을 불러와서 사용하고 종료 시 자동으로 종료가 되겠습니다. <br>그리고 FastAPI lifespan파라미터에 해당 함수를 적용해주면 됩니다.<br>

모델을 추론할 때마다 불러오면 상당히 비효율적이고 실제로도 속도 저하와 메모리 낭비가 이어지게 됨으로 한번만 불러오는 것이 옳은 방법입니다. <br>저는 예전에 이런줄도 모르고 모델을 계속 불러와서 사용하다가 테스트 당시 프로그램이 뻗어 왜 그런지 찾다가 알아냈네요... <br>
이제 HTTP 요청을 받아 FastAPI에게 전달하고 모델을 추론해서 결과를 다시 되돌려줘야하잖아요?<br>이 역할을 담당할 uvicorn이라는 녀석을 실행해줘야합니다. 

## 🔍 Uvicron 설치 및 실행
>uvicorn이라는 녀석은 python에서 ASGI(Asynchronous Server Gateway Interface)를 위한 경량 서버 구현체로 비동기 프로그래밍을 지원하는 녀석이라고 합니다. 나중에 다시 알아봐야겠어요...

```bash
pip install "uvicorn[standard]"
```
> uvicorn을 standard로 설치하면 성능 패키지도 같이 설치된다고 하네요. C로 구현한 패키지들이 설치가 되어 좀 더 빠르답니다. Claude 님께서 그렇게 얘기하시네요

설치가 완료되었다면 uvicorn을 실행시켜봅니다.
```bash
# 기본 실행, 아이피와 포트번호 지정 가능합니다.
uvicorn main:app --host 0.0.0.0 --port 8000 

# --reload 붙일 시 개발 모드입니다. 코드가 바뀌면 바로 수정됩니다.
uvicorn main:app --reload
```

실행 후 HTTP 요청을 보내어 결과를 확인해줍니다. 저는 post 방식 /predict 로 요청해서 받는걸로 정의를 했기에 여기로 요청을 보내보겠습니다.
```python
import numpy as np
import pandas as pd
from pathlib import Path
import requests

# 여기서 파일 경로를 지정합니다. 예시는 그냥 막 적은거에요
file_path = "D:/sdfsdf/sdfsd/sdfsdf/data.csv"
df = pd.read_csv(file_path, index_col=False)

# 모델에 맞게 데이터 처리는 해주셔야 합니다.
data = df.values.astype(np.float32)

# 
response = requests.post(
    """
    현재 로컬에서 uvicorn을 실행했으니 로컬 아이피인 127.0.0.1과 포트번호 적고 
    /predict 로 데이터를 보내 요청해보겠습니다.
    """
    "http://127.0.0.1:8000/predict",
    json={"data":data.tolist()}
)

result = response.json()

print(result)
```
코드 결과
```
{'recon_error': 0.0003860823344439268, 'threshold': 0.001415622653439641, 'is_anomaly': False, 'status': 'normal'}
```

자 정상적으로 모델 추론 결과를 돌려받았습니다. <br>원래 저희 회사는 C#인데 전 잘 몰라요 C#... 
아무튼 잡소리였구요. <br>
여기서 하나 더 나아가서 배포를 해야하는데 고객사 PC 장비에 파이썬 깔구, 패키지 깔구,... 배포하기 좀 힘들잖아요?? <br>그래서 그냥 실행파일로 서버를 배포하는것이 좋겠다 해서 pyinstaller를 알아봤습니다.

## 🔍 Pyinstaller
> 파이썬 스크립트(.py)를 윈도우, 맥, 리눅스에서 별도 설치 없이 실행 가능한 독립적인 실행파일(.exe)로 변환해주는 패키지입니다.

네. 그냥 실행파일만 만들어서 갔다 주고 실행하라고만 하면 됩니다. 완전 좋죠?
```bash
pip install pyinstaller
```
설치는 위와 같이 해주면 되고 코드에서 추가할 부분이 있습니다. main.py와 모델 불러오는 파일에 추가해줘야합니다.

```python
# 파일명 : main.py 

# -------- (생략) --------

# pyinstaller로 빌드하기 위해 main 실행 조건
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
> pyinstaller를 빌드하려면 main.py 맨 아래에 실행 진입점을 추가해야합니다.

모델 불러오는 파일에는 뭘 추가해야하냐! 모델 파일을 못찾는 경우가 생깁니다. 
```python
# 파일명 : inference.py

from pathlib import Path        # 파일또는 폴더 경로를 객체로써 조작 및 처리가 가능한 라이브러리
import sys
import numpy as np
import torch
import torch.nn as nn

from logger import logger

# 모델 클래스 불러오기
from models.CNNAutoencoder.model import CNNAutoEncoder

"""
# getattr(object, attribute, default)
[파라미터]
- object    : 필수값, 객체
- attribute : 필수값, object(객체)의 속성명
- default   : 선택값, object의 attribute 속성이 없다면 반환할 값 지정

[설명]
sys.frozen 속성이 있으면 pyinstaller로 빌드된 exe로 실행중이라는 뜻입니다.
일반 python으로 실행할 경우는 frozen 속성이 없기에 False를 반환하게 하였습니다.
"""
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent      # 모델 경로 지정을 위한 기본 경로
    logger.info(f"BASE_DIR: {BASE_DIR}")
else:
    BASE_DIR = Path(__file__).parent
    logger.info(f"BASE_DIR: {BASE_DIR}")

# pathlib을 사용하면 이와 같이 문자열 대신 객체로 경로를 다뤄서 / 연산자로 경로를 이어붙일 수 있습니다.
MODEL_PATH = BASE_DIR / "cnn_ae_fin.pt"     # 모델 경로 지정

# 실행시켰을 때 어느 경로로 모델 파일을 보고 있는지 확인하기 위한 logger
logger.info(f"MODEL_PATH: {MODEL_PATH}")

# ----- (생략) ------
```
> getattr을 통해 sys에 frozen 속성이 있다면 Path 경로를 BASE_DIR = Path(sys.executable).parent 이와 같이 정해주어 모델 파일을 잘 찾을 수 있도록 지정합니다.

코드를 수정 완료 했다면 이제 빌드를 진행해보겠습니다.
```bash
# 빌드할 파일 경로로 들어가 줍니다. 저의 경우 C:\fastapi에 있습니다.
cd C:\fastapi
# --onefile 옵션을 설정하면 모든 걸 exe 하나로 묶습니다.
# --name 은 생성될 exe 파일이름입니다.
pyinstaller --onefile --name cnn_ae_server main.py
```
> 빌드 명령어 실행 시 다소 시간이 걸릴 수 있으며, 완료될 경우 dist라는 폴더에 실제 빌드된 파일(.exe)이 생성됩니다. <br>배포 진행 시 모델 파일과 같이 전달하며 모델을 찾는 경로에 알맞게 파일을 넣어주시면 됩니다.

## ⚠️ 주의 사항
> 빌드는 반드시 windows에서 해야 windows용 exe가 나옵니다. <br>
백신이 exe를 오탐할 수 있습니다. --> 고객사에게 예외처리 요청하거나 검사 대상에서 제외합니다.<br>8000번 포트 방화벽 허용 필요할 수 있습니다.

## 💻 실행 결과
![Fastapi 실행화면](/assets/img/python/fastapi_pyinstaller_exe.png)
> 위와 같이 정상 실행되었다면 HTTP 요청으로 테스트 해보면 되겠습니다.

## 🔚 마치며(잡소리)
FastAPI를 통해 Model Serving을 할 수 있는 환경을 만들어봤습니다. 배포까지 쭉 왔네요.<br>
내용 자체는 너무 깊이 있게 다루진 않았다고 인지하고 있습니다. <br>저 같은 초보에게는 일단 어떻게 돌아가는지만 알면 그 다음 단계는 비교적 쉽게 이해할 수 있을테니까요. <br>어떤 글을 보면 엄청 딥하게 들어가는 글이 있던데 저의 수준으로는 이해하기 힘들뿐더러 집중도 안되더라구요...<br>
아무튼 공부한 내용을 하나하나 글을 올리면서 복습도 되는 것 같아 참 괜찮네요!! <br>시간이 좀 걸려서 그렇지 ㅋㅋ 이만 들어가보겠습니다!! <br>
아!! 코드는 따로 제 깃허브 페이지 링크에 올리겠습니다!!!

[코드 파일] <https://github.com/root0615/root0615.github.io/tree/main/assets/files/fastapi_model_serving>


## 오늘도 화이팅하세요!! 응원하겠습니다!!












