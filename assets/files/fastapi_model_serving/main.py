import time
# 서버 시작/종료 시 실행할 코드를 정의하기 위한 데코레이터
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
# 요청/응답 스키마 정의할 때 사용
from pydantic import BaseModel

from logger import logger
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
        위와 같이 저장하면 서버가 살아있는 동안 계속 유지되기에 해당방법으로 사용했습니다.
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

# pyinstaller로 빌드하기 위해 main 실행 조건
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)