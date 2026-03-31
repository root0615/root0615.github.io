import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)    # 없으면 폴더 자동 생성

def setup_logger(name: str = "app") -> logging.Logger:
    # name으로 logger를 가져옵니다. 같은 이름으로 다시 호출하면 새로 만들지 않고 기존 것을 반환합니다.(=싱글톤 패턴)
    logger = logging.getLogger(name)
    """
    # logger.setLevel()
    logger가 처리할 최소 로그 레벨을 설정합니다. 'DEBUG'로 설정 시 모든 레벨의 로그를 처리합니다.
    
    [로그 레벨 순서]
    DEBUG(개발 시 상세 정보) < INFO(일반 정보) < WARNING(경고) < ERROR(에러) < CRITICAL(심각 에러)
    """
    logger.setLevel(logging.DEBUG)

    """
    # logging.Formatter()
    로그 출력 형식을 정의합니다.
    - `%(asctime)s` : 시간
    - `%(levelname)s` : 로그 레벨 (INFO, ERROR 등)
    - `%(message)s` : 실제 로그 메시지
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러 (터미널에 로그를 출력하는 핸들러)
    console_handler = logging.StreamHandler()   # 콘솔 출력 담당
    console_handler.setLevel(logging.INFO)      # INFO로 레벨 설정
    console_handler.setFormatter(formatter)     # 위에서 정의한 포맷 적용

    # 파일 핸들러 (로그 파일 저장용 핸들러)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log"     # 로그 저장 경로 설정
    file_handler = logging.FileHandler(log_file, encoding="utf-8")      # 파일 저장 담당, encoding으로 한글 깨짐 방지
    file_handler.setLevel(logging.DEBUG)    # DEBUG로 레벨 설정
    file_handler.setFormatter(formatter)    # 위에서 정의한 포맷 적용

    # logger에 핸들러를 등록합니다.
    # 콘솔, 파일 핸들러 2개 모두 등록하여 로그 하나가 기록될 때 콘솔과 파일 모두 기록됩니다.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 설정완료된 logger를 반환합니다.
    return logger

# 전역 logger
# 모듈이 import되었을 때 파일 전체를 한번 실행하게 되며 logger를 sys.modules에 캐싱합니다.
# 그래서 어느 파일에서 로그를 찍든 같은 파일에 로그가 쌓입니다.
logger = setup_logger()

"""
# 만약에 파일별로 로그를 따로 찍고 싶으면??
각 파일마다 setup_logger를 import하여 name을 다르게 주면 됩니다.
물론 저장할때 로그파일명에 name을 추가하면 따로 저장이 됩니다.

from logger import setup_logger
logger = setup_logger(name="main")

from logger import setup_logger
logger = setup_logger(name="inference")
"""
