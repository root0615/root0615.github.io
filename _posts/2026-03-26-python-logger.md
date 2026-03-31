---
title: "[python] print 말고 logging 사용해볼까?"
date: 2026-03-26
categories: [python]
tags: [logging, 파이썬로그]
---

## 🔍 logging은 파이썬 내장 로깅 도구
> logging은 파이썬의 표준 라이브러리 중 하나로, 애플리케이션에서 발생하는 이벤트를 추적하기 위한 유연하고 확장 가능한 로깅 시스템을 제공합니다. 개발 과정에서나 배포된 애플리케이션에서 발생하는 정보, 경고, 오류 등을 기록하는 데 사용됩니다. print 문 대신 logging을 사용하면, 로그의 중요도에 따라 다른 동작을 정의할 수 있고, 로그를 콘솔, 파일, 네트워크 서버 등 다양한 대상으로 쉽게 리디렉션할 수 있습니다.<br>
[출처] <https://wikidocs.net/236287>

음...그렇다고 합니다. 그냥 간단하게 값이 뭐가 나오는지 확인할 때마다 print만 주구장창 쓰다가 뭔가 로그 기록해야한다고 하니 찾아보게 되었습니다. 확실히 구현해놓고 사용하니 하길 잘했다 생각합니다.<br>
코드를 한번 작성해보았습니다. 역시 Claude의 도움을 200% 받았으며, 사실과 다를 수 있습니다.<br>
저는 완전 초보중 바닥이며, 저의 스승은 Claude와 블로그 선생님분들이십니다. (관심없겠지만 그렇다고요...)

## 📄 코드 
```python
# 파일명: logger.py

import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str = "app") -> logging.Logger:
    # 이름별 싱글톤 패턴
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
    console_handler = logging.StreamHandler()   # 콘솔 출력 담당 핸들러
    console_handler.setLevel(logging.INFO)      # INFO로 레벨 설정
    console_handler.setFormatter(formatter)     # 위에서 정의한 formatter 사용

    # 파일 핸들러 (로그 파일 저장용 핸들러)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log" # 파일 저장 경로
    file_handler = logging.FileHandler(log_file, encoding="utf-8")  # 파일 저장 담당
    file_handler.setLevel(logging.DEBUG)    # DEBUG 레벨 설정
    file_handler.setFormatter(formatter)    # 동일하게 위의 formatter 사용

    # logger에 핸들러를 등록합니다.
    # 콘솔, 파일 핸들러 2개 모두 등록하여 로그 하나가 기록될 때 모두 기록됩니다.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 설정완료된 logger를 반환합니다.
    return logger

# 전역 logger
logger = setup_logger()
```

## 📝 코드에 대한 부가 설명
```python
def setup_logger(name: str = "app") -> logging.Logger:
    # 이름별 싱글톤 패턴
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

# --- 생략 ---

# 전역 logger
logger = setup_logger()
```
> logger = logging.getLoger(name) 에서 name으로 logger를 가져옵니다. 같은 이름으로 다시 호출하면 **새로 만들지 않고 기존 것을 반환하는 싱글톤 패턴**을 사용합니다.<br>
logger가 import 되었을 때 파일 전체를 한번 실행하게 되어 logger를 sys.modules에 캐싱합니다. 그래서 어느 파일에서든지 import해서 로그를 찍으면 같은 파일에 로그가 쌓이게 됩니다. 조건은 **같은 이름**으로 했을 경우입니다.<br>
그런데 만약 파일별로 로그를 따로 찍고 싶으면?<br>
setup_logger를 import 하여 name을 다르게 주면 됩니다. 물론 파일을 저장한다면 로그파일명에 name을 따로 표시해야겠지요.<br>
>> <예시><br>
#main.py 파일<br>
from logger import setup_logger<br>
logger = setup_logger(name="main")<br><br>
#inference.py 파일<br>
from logger import setup_logger<br>
logger = setup_logger(name="inference")<br>
이런식으로 진행하면 됩니다.


## 📄 코드 사용 예시
```python
from logger import logger

logger.debug("DEBUG 레벨로 로그 찍힘")
logger.info("INFO 레벨로 로그 찍힘")
logger.warning("WARNING 레벨로 로그 찍힘")
logger.error("ERROR 레벨로 로그 찍힘")
logger.critical("CRITICAL 치명적 로그 찍힘")
```

## 💻 Console 코드 결과
```
[2026-03-26 16:55:55] [INFO] INFO 레벨로 로그 찍힘
[2026-03-26 16:55:55] [WARNING] WARNING 레벨로 로그 찍힘
[2026-03-26 16:55:55] [ERROR] ERROR 레벨로 로그 찍힘
[2026-03-26 16:55:55] [CRITICAL] CRITICAL 치명적 로그 찍힘
```
뭔가 이상한데? DEBUG 로그 어따 팔아먹었지??<br>
... 네, 코드상에 살펴보면 콘솔 최소 로그 레벨은 INFO라서 그렇습니다. 예 ...<br>
console_handler.setLevel(logging.INFO)<br>
그래서 안나왔어요... 순간 뭔가하고 당황해서 한참 살펴봤습니다... 내가 해놓고 내가 모르네<br>
파일로 저장한건 잘 저장되었습니다. 확인했어요!

## 🔚 마치며(잡소리)
와 블로그 글 쓰는데 뭐가 이렇게 시간을 많이 소비하지?? 때려칠까?? 하는 생각이 문득 드네요 고작 2개 올렸는데 벌써 접어버리고 싶네요... 진짜 꾸준히 올리시는 분들 존경스럽습니다. 나중에 빨라지려나??<br>
좀 참고 공부한다고 생각하고 올려보겠습니다. 여유 있을때요... 여유...<br>
아 그리고 logging은 요새 많이 안쓰고 loguru?? 그거 쓴다고 합니다. 네. 공부해야겠어요.

**오늘도 이리 치이고 저리 치이느라 고생하셨습니다.**