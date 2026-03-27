---
title: "[python] matplotlib에서 mplcursors를 사용해서 포인트 팝업창 띄우기"
date: 2026-03-25
categories: [python]
tags: [matplotlib, mplcursors]
---

### 💬잡소리
이 게시물이 처음 블로그에 작성하는 글이 되었습니다. 솔직히 요즘 AI로 인해서 블로그 보는 일이 많진 않겠지만 그래도 제가 공부하고 확인된 내용을 올리는게 좋은 습관이 될것 같아 글을 올려보려고 합니다. 솔직히 이 게시물이 처음이자 마지막이 될 수도 있어요 ㅋㅋㅋㅋ... 아무튼 해보겠습니다. 아! 그리고 글의 내용은 사실과 다를 수 있음을 인지하고 참고만 해주시면 감사하겠습니다. 혹시 모르잖아요 갑자기 안될 수도 있으니까 최대한 제가 테스트해보고 나온 결과를 올리긴 할거에요 하핳. 그럼 본론으로 넘어가겠습니다.


## 🔍mplcursors는 뭘 해주는 라이브러리인가?
>(Claude 왈) 'matplotlib'으로 만든 그래프에 마우스로 클릭하면 정보가 팝업으로 뜨게 해주는 라이브러리이다.<br>

Matlab 그래프를 그려보셨다면 알겠지만 plot(data)로 실행하면 
![Matlab figure 그래프](/assets/img/matlab/matlab_figure.png)
이렇게 포인트를 찍어 값을 확인할 수 있습니다.<br>
plot()만!! 해서 실행하면 이렇게 찍을 수 있는데!! 파이썬은 안되더라구요...?

그래서 찾아봐서 해결한 것이 **mplcursors**였습니다.<br>
## 라이브러리 설치
```bash
pip install mplcursors
```

## 📄 코드 
```python
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
%matplotlib qt5 

x = [1, 2, 3, 4, 5]
y = [10, 25, 15, 30, 20]

fig, ax = plt.subplots()    # 액자 전체 (fig), 그리려고 하는 칸 (ax)
sc = ax.scatter(x, y)       # ax를 sc변수에 저장, mplcursors에게 보내려고 저장함.
"""
# sc 변수를 mplcursors에게 커서 달아달라 하기
# sc : 커서를 달 대상 그래프
# mltiple=True : 여러점을 동시에 클릭해서 팝업 여러개 띄울 수 있게 허용

- 파라미터들
hover=True      : 마우스 올리기만 해도 표시
multiple=True   : 여러점 동시에 선택가능
highlight=True  : 선택된 점을 강조표시
"""
mplcursors.cursor(sc, multiple=True)
plt.show()
```

## 💻 코드 결과
![matplotlib에 mplcursors 적용한 그래프](/assets/img/python/mplcursor_example.png)

정말 간단한 코드 한줄로 이렇게 포인트를 클릭하면 값이 출력되게 나왔습니다.<br>
사용법도 간단합니다.
- 포인트 왼쪽 클릭 : 포인트 팝업창 띄우기<br>
- 팝업창 오른쪽 클릭 : 팝업창 지우기

## 🔚 마치며
Matlab에서 그래프 보다가 Python에서 보고 싶은데 뭔가 만족스럽지 못해서 찾아봤습니다.<br>
가끔은 python에서 실행해야할 때가 있었는데 그래프 보기 너무 불편하더라구요.<br>
아무튼 누가 과연 볼까? 싶은 게시글이긴 하지만 그냥 블로그로 정리하면서 복습도 되고 좋은것 같네요. 다음 글이 나올지는 모르겠습니다. 하하... <br>

## 오늘도 고생하셨습니다.
