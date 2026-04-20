---
title: "[Python] LLM, RAG 이해하기 쉽게 접근해보자 2편"
date: 2026-04-20
categories: [Python]
tags: [LLM, RAG, Chunking, Chromadb, SentenceTransformer]
---

## 💬 인삿말(잡소리)
LLM에 이어서 RAG로 넘어가보겠습니다.<br>
1편에서 봤다시피, RAG는 LLM 모델이 학습하지 못한 부분을 내가 직접 데이터를 구해서 학습하는거라고 말씀드렸습니다.<br>
그래서 오늘은 이상탐지 모델의 관련 텍스트를 데이터베이스에 넣고 질문하면, 관련 내용을 찾아서 LLM이 답하는 방법을 간단하게 해보겠습니다.<br>

[1편]<https://root0615.github.io/posts/python-LMM-RAG1/><br>

### ⚠️ 주의 사항 - 클로드랑 한거임
책이나 문서가 아닌 클로드와 주고 받은거를 정리한 것이기에 틀린 부분이 있을 수 있다는 점 참고하시기 바랍니다.<br>
왠만하면 제가 코드를 돌려가면서 테스트를 하며 진행했지만 내용 측면에서 틀린 부분이 있을 수도...<br>

## 🔍 RAG(Retrieval-Augmented Generation, 검색 증강 생성)이 무엇이냐?
> LLM의 한계를 보완하기 위해, 외부 데이터를 검색해서 LLM에게 제공하는 기법입니다.

📝 RAG 작동 흐름
```
1. 문서 준비        논문, PDF, txt 등 → 텍스트 추출
        ↓
2. 청킹 (Chunking)  긴 문서를 작은 조각으로 자르기
        ↓
3. 임베딩           각 조각을 숫자 벡터로 변환
        ↓
4. 벡터 DB 저장     ChromaDB 같은 DB에 저장
        ↓
5. 검색 (Retrieve)  질문과 가장 유사한 조각 찾기
        ↓
6. 생성 (Generate)  찾은 조각 + 질문을 LLM에 넘겨서 답변 생성
```

### 💾 패키지 설치
```python
# chromadb: 벡터를 저장하고 검색하는 DB
# sentence-transformers: 텍스트를 숫자 벡터로 변환하는 임베딩 모델
pip install chromadb sentence-transformers
```

### 📄 RAG 테스트 코드
```python
from groq import Groq           # LLM 한테 질문하는 도구
import chromadb                 # 벡터를 저장하고 검색하는 DB
from sentence_transformers import SentenceTransformer   # 텍스트를 숫자 벡터로 변환하는 임베딩 모델

# 1. 준비
groq_client = Groq(api_key="Groq API 키 여기 넣어야하는데 복사해놨죠??")
chroma_client = chromadb.Client()
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 임베딩 모델
"""
임베딩이 뭐냐 
- 텍스트를 숫자 벡터로 바꾸는 것. 
의미가 비슷한 문장은 벡터공간에서 가까운 위치에 놓인다.
이상탐지에서 feature vector를 만드든 것과 개념이 같다.

Chroma DB는 뭔데
- 이 벡터들을 저장하고 "가장 가까운 벡터 찾아줘" 검색을 해주는 DB

흐름 요약
질문 -> 임베딩 -> 유사 문서 검색 -> (LLM에 문서 + 질문 전달) -> 답변
"""

# 2. 문서 준비 (LLM이 참고할 지식 창고. 실제로는 PDF나 파일에서 읽어오면 된다.)
documents = [
    "Isolation Forest는 데이터를 무작위로 분리하여 이상치를 탐지한다. 이상치일수록 적은 분리 횟수로 고립된다.",
    "OCSVM은 정상 데이터의 경계를 학습하고, 경계 밖의 데이터를 이상치로 판단한다.",
    "LSTM Autoencoder는 시계열 데이터를 압축했다가 복원하며, 복원 오차가 크면 이상치로 판단한다.",
    "anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.",
    "이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.",
]

# 3. 임베딩 후 ChromaDB에 저장(핵심 내용)
# create_collection : 서랍장 하나 만들기. 나중에 문서들을 여기에 넣음
collection = chroma_client.create_collection("anomaly_docs")

"""
# embedder.encode(documents) : 텍스트를 숫자 벡터로 변환
"Isolation Forest는..." → [0.23, -0.45, 0.87, 0.12, ...]
"OCSVM은..."           → [0.31, -0.41, 0.79, 0.08, ...]

의미가 비슷한 문장일수록 벡터값이 비슷하게 나온다.
이상치 탐지에서 feature vector를 만드는 것과 같은 개념이다.
"""
embeddings = embedder.encode(documents).tolist()

# collection.add : 문서 원본 + 벡터 + ID를 DB에 저장. ID는 나중에 구분하기 위한 이름표
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("문서 저장 완료")

# 4. 질문 -> 검색 -> 생성
def rag_chat(question):
    # 질문을 임베딩, 질문도 똑같이 벡터로 변환해야 문서 벡터들과 비교가 가능
    query_embedding = embedder.encode([question]).tolist()
    
    """
    # 유사한 문서 검색
    질문 벡터와 가장 가까운 문서 벡터 2개를 찾는다.
    벡터 간의 거리를 계산해서 "의미적으로 가장 비슷한" 문서를 골라주는 것

    질문 벡터   "anomaly score -0.35"
    ↓ 거리 계산
    문서1 벡터  "anomaly score 음수일수록..."  → 거리 0.12 (가장 가까움✅)
    문서2 벡터  "threshold 설정..."           → 거리 0.18 (두번째✅)
    문서3 벡터  "OCSVM은..."                  → 거리 0.67 (멀어서 제외)
    
    results로 나온 결과는 dictionary 타입으로 나오고
    documents 키에 값이 있는데 n_results=2개만 찾는거니 2개만 있겠지?
    그리고 가장 질문벡터와 거리가 짧은(비슷한) 문서는 첫번째(0)인덱스에 있는 것을 가져온 것이다.
    """
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    retrieved_docs = results["documents"][0]
    
    
    """
    # 검색된 문서를 컨텍스트로 만들기
    
    위의 retrieved_docs는 검색된 문서들이 리스트 형태로 담겨있다.
    예시)
    retrieved_docs = [
        "anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.",
        "이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 false positive가 증가한다."
    ]
    
    이걸 그대로 LLM한테 넘기면 리스트형태라 보기 불편
    그래서 "\n".join()으로 줄바꿈으로 이어 붙여서 하나의 텍스트로 만든다.
    """
    context = "\n".join(retrieved_docs)
    # 아래 print는 잘 검색되었는지 확인하기 위한 디버깅용
    print(f"\n검색된 문서:\n{context}\n")
    
    """
    # LMM에게 전달
    검색된 문서를 system 프롬프트 안에 넣어서 LLM에게 넘긴다. 
    LLM 입장에서는 "나한테 이미 관련 자료가 주어졌으니 이걸 참고해서 답하면 되겠다."가 된다.
    """
    message = [
        {
            "role": "system",
            "content": "너는 이상탐지 전문가야. 아래 참고 문서를 바탕으로 질문에 답해줘.\n\n"
                        f"[참고문서]\n{context}"
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
    )
    
    return response.choices[0].message.content

# 실행
answer = rag_chat("anomaly score가 -0.35면 이상치야?")
print(f"AI: {answer}")
```
### 💻 출력 결과
```

나: anomaly score가 -0.35면 이상치야?

-----------------------------------------------

검색된 문서:
anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.
이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.

AI: 아니. 문서에 따르면, anomaly score가 -0.5 이하일 때는 강한 이상 신호로 볼 수 있다고 한다. 따라서 -0.35는 아직 -0.5를 넘지 못했으므로, 강한 이상 신호는 아니다. 다만 음수이므로 이상치일 가능성은 있다.
```

### 📝 RAG 흐름 다시 전체 정리
```
# 전체 흐름 
[사전 작업]
문서들 → 임베딩 → ChromaDB 저장

[질문이 들어올 때마다]
질문 → 임베딩 → 유사 문서 검색 → LLM에 (문서 + 질문) 전달 → 답변

# 일반 LLM 과 RAG 차이

## 일반 LLM
messages = [
    {"role": "system", "content": "너는 전문가야."},
    {"role": "user",   "content": "anomaly score -0.35는?"}
]
### LLM이 자신의 고유 정보만으로 답함 → 내 도메인 지식 없음

## RAG
messages = [
    {"role": "system", "content": "너는 전문가야.\n[참고문서]\n내 문서 내용..."},
    {"role": "user",   "content": "anomaly score -0.35는?"}
]
### LLM이 내가 넣은 문서를 참고해서 답함 → 내 도메인 지식 반영 ✅

- 결론 : 결국 RAG의 핵심은 "LLM한테 질문하기 전에, 관련 문서를 찾아서 같이 넘겨주는 것" 이다.
```

## 🔍 문서파일 불러오기
우리가 LLM에게 알려줄 문서는 위와같이 짧은 내용이 아닌 엄청 많은 내용이 있겠죠.<br>
그래서 문서 파일을 불러와서 가져와야 합니다.<br>
이때 문서파일은 3가지 파일 형태를 가장 많이 씁니다.<br>
- 📄 TXT 파일 (가장 간단)
```python
with open("documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 통째로 넣거나
documents = [text]  
# 또는 줄 단위로 자르기
documents = text.split("\n")
```
- 📄 PDF 파일 (가장 흔함)
```python
# pip install pypdf
from pypdf import PdfReader

reader = PdfReader("documents.pdf")

# 페이지 단위로 나눠서 리스트에 담기
documents = []
for page in reader.pages:
    text = page.extract_text()
    # 빈 페이지 제외하고 있는 페이지만 추가
    if text:  
        documents.append(text)
```
- 📄 CSV 파일 (데이터 관련 문서)
```python
import pandas as pd

df = pd.read_csv("documents.csv", encoding="euc-kr")

documents = df.apply(
    lambda row: " ".join(row.astype(str)), axis=1
).tolist()
```

## 🔍 Chunking(청킹)
파일들은 위와 같이 불러오면 되는데 문제는 이걸 통째로 넣으면 안된다는 것입니다.<br>
문서가 너무 길면 임베딩 모델이 처리 못하거나 검색 정확도가 떨어집니다.<br>
그래서 `적당한 길이로 잘라내는 과정`이 중요한데 이러한 작업을 하는 것을 `Chunking(청킹)`이라고 한답니다.<br>
RAG 성능의 절반은 청킹을 얼마나 잘 하느냐에 달려있다고 할 만큼 중요한 개념이라고 합니다.<br>
왜 중요하냐면, `검색 정확도가 청킹에 따라 크게 달라지기 때문`입니다.<br>

```
질문: "Isolation Forest의 anomaly score 해석법은?"

청크가 너무 크면 (1000자):
→ 검색된 청크 안에 Isolation Forest, OCSVM, LSTM 내용이 다 섞여있음
→ LLM이 어떤 내용을 참고해야 할지 혼란

청크가 너무 작으면 (50자):
→ "Isolation Forest는 이상치를" 처럼 문맥이 잘려버림
→ LLM이 참고할 내용이 불충분

청크가 적당하면 (300자):
→ Isolation Forest 관련 내용만 딱 담김
→ LLM이 정확하게 참고 가능 ✅

# 실무 팁
청킹 사이즈는 문서 종류에 따라 다르게 잡습니다.
논문/기술문서  → 500~1000자  (내용이 밀도 높음)
뉴스/블로그    → 300~500자   (단락이 짧음)
FAQ 문서       → 질문+답변 1세트를 하나의 청크로
매뉴얼         → 섹션 단위로

# 결론
청킹은 RAG를 실제로 잘 작동하게 만드는 핵심 튜닝 포인트입니다.. 
나중에 RAG 성능이 기대보다 안 나온다 싶으면 청킹 전략을 바꾸는 게 가장 먼저 해볼 것 중 하나예요.
```

### 📝 Chunking의 3가지 전략
|방법|내용|
|---|---|
|고정 크기 청킹|글자 수로 자르기.<br>가장 단순하지만 문장이 어색하게 잘릴 수 있음.|
|문장/단락 기준 청킹|문장이나 문단 단위로 자르기.<br>문맥이 자연스럽게 유지됩니다.<br>실무에서 가장 많이 사용하는 방법이랍니다.<br>(Claude가 한말이라 아닐 수도 있어요!)|
|시맨틱 청킹|의미가 바뀌는 지점을 AI가 판단해서 자르기.<br>가정 정확하지만 구현이 복잡하다.|

방법들을 하나씩 코드로 살펴보겠습니다.
### 📄 고정 크기 청킹
```python
from pypdf import PdfReader

# 문서 파일을 PDF 파일로 불러오는 함수를 사용해보겠습니다.
def load_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        # 페이지가 존재하면 추가
        if text:
            full_text.append(text)
    
    # 리스트 형태이니 마지막에 \n을 이어 붙여서 텍스트 형태로 변환
    result = "\n".join(full_text)
    return result

# 고정크기 청킹 함수(overlap에 관하여 아래 설명)
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

text = load_pdf("documents.pdf")
documents = chunk_text(text)
print(documents)
print(f"총 {len(documents)}개 청크로 나뉨")
```
> 📌 고정 크기 청킹 시 Overlap이 필요한 이유<br>
문맥이 청크 경계에서 잘리는 걸 방지하기 위해<br>
<br>
overlap 없이 자르면:<br>
청크1: "...Isolation Forest는 이상치를"<br>
청크2: "탐지하는 알고리즘이다..."  ← 앞뒤 문맥 없음<br>
<br>
overlap 있으면:<br>
청크1: "...Isolation Forest는 이상치를"<br>
청크2: "이상치를 탐지하는 알고리즘이다..."  ← 앞 내용 살짝 겹침 ✅<br>

💻 코드 결과
```
[
    'Isolation Forest는 데이터를 무작위로 분리하여 이상치를 탐지한다. 이상치일수록 적은 분리 횟수로 고립된다.\nOCSVM은 정상 데이터의 경계를 학습하고, 경계 밖의 데이',
    '의 경계를 학습하고, 경계 밖의 데이터를 이상치로 판단한다.\nLSTM Autoencoder는 시계열 데이터를 압축했다가 복원하며, 복원 오차가 크면 이상치로 판단한다.\nanomal',
    '크면 이상치로 판단한다.\nanomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.\n이상탐지에서 threshold 설정은 매우 ',
    '에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.'
]

총 4개 청크로 나뉨
```

### 📄 문장/단락 기준 청킹
```python
import re

def chunk_by_sentence(text, chunk_size=3):
    # 문장 단위로 자르기(. ! ? 기준)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 빈 문자열 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        
        # chunk_size 문장이 모이면 하나의 청크로
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        
    # 남은 문장 처리
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def chunk_by_paragraph(text):
    # 단락 단위로 자르기(빈 줄 기준)
    paragraphs = text.split("\n\n")
    
    # 빈 단락 제거
    chunks = [p.strip() for p in paragraphs if p.strip()]
    
    return chunks

# 청킹 작업할 text
text = """
Isolation Forest는 데이터를 무작위로 분리하여 이상치를 탐지한다. 이상치일수록 적은 분리 횟수로 고립된다.
OCSVM은 정상 데이터의 경계를 학습하고, 경계 밖의 데이터를 이상치로 판단한다.
LSTM Autoencoder는 시계열 데이터를 압축했다가 복원하며, 복원 오차가 크면 이상치로 판단한다.

anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.
이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.
"""

# 2문장씩 묶어서 청킹
sentence_result = chunk_by_sentence(text, chunk_size=2)
# 단락 단위로 청킹
paragraph_result = chunk_by_paragraph(text)

print(sentence_result)
print(len(sentence_result))
print()
print(paragraph_result)
print(len(paragraph_result))
```
💻 코드 결과
```
[
    'Isolation Forest는 데이터를 무작위로 분리하여 이상치를 탐지한다. 이상치일수록 적은 분리 횟수로 고립된다.',
    'OCSVM은 정상 데이터의 경계를 학습하고, 경계 밖의 데이터를 이상치로 판단한다. LSTM Autoencoder는 시계열 데이터를 압축했다가 복원하며, 복원 오차가 크면 이상치로 판단한다.',
    'anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.',
    '이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.'
]

4

[
    'Isolation Forest는 데이터를 무작위로 분리하여 이상치를 탐지한다. 이상치일수록 적은 분리 횟수로 고립된다.\nOCSVM은 정상 데이터의 경계를 학습하고, 경계 밖의 데이터를 이상치로 판단한다.\nLSTM Autoencoder는 시계열 데이터를 압축했다가 복원하며, 복원 오차가 크면 이상치로 판단한다.', 
    'anomaly score가 음수일수록 이상치일 가능성이 높다. -0.5 이하면 강한 이상 신호로 볼 수 있다.\n이상탐지에서 threshold 설정은 매우 중요하다. 너무 낮으면 정상을 이상으로 탐지하는 false positive가 증가한다.'
]

2
```

### 📄 시맨틱 청킹 
```python
from sentence_transformers import SentenceTransformer
"""
# cosine_similarity (코사인 유사도)
두 벡터가 얼마나 같은 방향을 가르키는지 측정하는 지표.
값은 -1 ~ 1 사이

1.0   → 완전히 같은 방향 (의미가 매우 유사)
0.0   → 90도 방향 (관련 없음)
-1.0  → 완전히 반대 방향 (의미가 반대)
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

embedder = SentenceTransformer("all-MiniLM-L6-v2")

#%%
def chunk_by_semantic(text, threshold=0.7):
    # 문장 단위로 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 빈 문자열 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 각 문장을 임베딩
    embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # 이전 문장과 현재 문장의 코사인 유사도 계산
        similarity = cosine_similarity(
            [embeddings[i-1]],      # 이전 문장 벡터
            [embeddings[i]]         # 현재 문장 벡터
        )[0][0]
        """
        뒤에 [0][0] 쓰는 이유?
        결과가 2D 배열처럼 나옴. -> [[1.0]]
        여러벡터를 한번에 비교할 수 있는 구조이기 때문
        """

        # 유사도 출력
        print(f"'{sentences[i-1][:20]}...' vs '{sentences[i][:20]}...' → 유사도: {similarity:.3f}")
        
        # 지정한 threshold를 기준으로 같은주제인지 아닌지 구별
        if similarity >= threshold:
            # 유사도 높음 -> 같은 주제 -> 같은 청크에 합치기
            current_chunk.append(sentences[i])
        else:
            # 유사도 낮음 -> 주제 바뀜 -> 새 청크 시작
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks
            
# 사용
text = """
Isolation Forest는 데이터를 무작위로 분리한다. Isoaltion Forest는 이상치일수록 적은 횟수로 고립된다.
오늘 날씨는 매우 맑고 화창하다. 산책하기 좋은 화창한 날씨다.
OCSVM은 정상 데이터의 경계를 학습한다. OCSVM의 경계 밖의 데이터를 이상치로 판단한다.
"""

# 적절한 유사도 기준점(threshold)을 정해야한다.
chunks = chunk_by_semantic(text, threshold=0.6)
for i, chunk in enumerate(chunks):
    print(f"chunk {i+1}: {chunk}\n")
```
💻  코드 결과
```
'Isolation Forest는 데이...' vs 'Isoaltion Forest는 이상...' → 유사도: 0.718
'Isoaltion Forest는 이상...' vs '오늘 날씨는 매우 맑고 화창하다....' → 유사도: 0.230
'오늘 날씨는 매우 맑고 화창하다....' vs '산책하기 좋은 화창한 날씨다....' → 유사도: 0.906
'산책하기 좋은 화창한 날씨다....' vs 'OCSVM은 정상 데이터의 경계를 학...' → 유사도: 0.168
'OCSVM은 정상 데이터의 경계를 학...' vs 'OCSVM의 경계 밖의 데이터를 이상...' → 유사도: 0.989

chunk 1: Isolation Forest는 데이터를 무작위로 분리한다. Isoaltion Forest는 이상치일수록 적은 횟수로 고립된다.

chunk 2: 오늘 날씨는 매우 맑고 화창하다. 산책하기 좋은 화창한 날씨다.

chunk 3: OCSVM은 정상 데이터의 경계를 학습한다. OCSVM의 경계 밖의 데이터를 이상치로 판단한다.
```
> ⚠️ 위의 결과를 확인했을 때 문장의 유사도 값을 비교하여 적절한 threshold값을 정해줄 필요가 있다.

## 🔚 마치며(잡소리)
RAG와 chunking까지 알아봤습니다.<br>
나름 쉽게 이해할 수 있도록 적긴했습니다.<br>
저처럼 LLM이 뭐? RAG는 또 뭐야? 하시는 분들이 만약 있다면, 이해하는데 도움이 되었으면 좋겠습니다.<br>
오늘도 응원하겠습니다.<br>
화이팅입니다!!<br>

[1편]<https://root0615.github.io/posts/python-LMM-RAG1/><br>