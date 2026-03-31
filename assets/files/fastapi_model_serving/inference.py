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

logger.info(f"MODEL_PATH: {MODEL_PATH}")

# 모델 로드와 추론을 담당하는 클래스를 선언합니다.
class AnomalyDetector:

    # 클래스 생성 시 자동 실행되며, 바로 밑에 있는 _load() 메서드로 모델 로드합니다.
    def __init__(self, device:str="cpu"):       # device 기본값 cpu
        self.device = device
        self._load()

    # 모델 로드 메서드
    def _load(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다. {MODEL_PATH}")

        try:
            # 모델 파일을 로드
            """
            # weights_only 파라미터에 대해
            먼저 알아두어야할 사실은 .pt 파일은 내부적으로 python의 pickle 형식으로 저장됩니다.
            이게 위험할 수 있습니다. 왜 위험하냐? pickle 파일은 로드할 때 파일 안의 코드를 실행하는 구조입니다.
            만약 파일 안 코드가 악의적인 코드가 존재한다면 그냥 불러오는 순간 실행되어 악영향을 끼치게 됩니다.
            그러므로 weight_only=True로 불러오면 코드를 실행하지 않고 안전하게 가져올 수 있게 됩니다.
            """
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            # 모델 파일에서 dict 키를 이용하여 값 꺼내기
            self.num_feature = checkpoint["model_config"]["num_feature"]
            self.target_len = checkpoint["model_config"]["target_len"]
            self.latent_dim = checkpoint["model_config"]["latent_dim"]

            # 모델을 변수에 저장, 이 시점은 랜덤 가중치 상태
            self.model = CNNAutoEncoder(self.num_feature, self.target_len, self.latent_dim)
            # 학습했던 가중치 값을 모델에 적용
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # 모델 판정모드로 전환 (학습모드는 model.train()으로 진행한다. 여기서는 이미 학습이 완료되었으니 eval())
            self.model.eval()

            # Z-score scale로 전처리 하였기에 저장해두었던 평균과 표준편차를 가져온다.
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]

            # 모델의 threshold(threshold를 기준으로 정상/이상 판정)를 가져온다.
            self.threshold = checkpoint["threshold"].item()

        # 예외처리
        except FileNotFoundError:
            raise
        except KeyError as e:
            raise KeyError(f"체크포인트에 필요한 키가 없음: {e}")
        except Exception as e:
            raise RuntimeError(f"모델 로드 중 오류 발생: {e}")

    # 추론 메서드(data를 받아서 처리한다)
    def predict(self, data: np.ndarray) -> dict:

        # z-score scale
        scaled = (data - self.mean) / self.std

        # 텐서 변환 및 배치 차원 추가
        x = torch.tensor(scaled, dtype=torch.float32)
        """
        data의 형태를 학습할 때 (Batch, Time, channel) 형태로 넣었기에 unsqueeze(0)으로 
        0번째 차원(batch)을 추가해준다.
        
        원래 CNNAutoencoder는 (Batch, channel, Time)으로 배치해서 학습해야합니다.
        그래서 모델 클래스 내부에서 변환했습니다. 그러므로 따로 channel과 Time을 전치하지 않고 집어넣습니다.
        """
        x = x.unsqueeze(0)

        """
        # 추론시에는 gradient 계산이 필요없기에 no_grad()로 감싸서 추론합니다.
        - gradient : 벡터 공간에서 스칼라 함수의 최대 증가율을 나타내는 벡터입니다.
        라는 정의가 있는데 영어 뜻 그대로 기울기입니다. 간단하게 생각하면
        torch.no_grad()는 기울기 계산을 안한다고 생각하면 될것 같습니다.
        그러니 메모리는 절약되고 속도도 향상되는 효과가 있습니다.
        """
        with torch.no_grad():
            recon, z = self.model(x)

        # 원본(data)과 모델에서 추론한 값의 차이를 구합니다. (재구성 오차 = MSE Loss)
        recon_error = ((x - recon)**2).mean(dim=[1, 2]).item()
        # threshold로 이상치인지 아닌지 판정 (True = 이상치, False = 정상)
        is_anomaly = recon_error >= self.threshold

        # 최종 결과를 dictionary로 반환합니다. 이걸 FastAPI가 JSON으로 변환해서 전달합니다.
        return {
            "recon_error":recon_error,
            "threshold":self.threshold,
            "is_anomaly":is_anomaly,
            "status":"normal" if recon_error < self.threshold else "anomaly",
        }






