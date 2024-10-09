# cuda_matmul
optimizing cuda matmul kernel from scratch. </br>
행렬곱 연산 커널을 최적화 하는 방법을 단계별로 구현하고 있습니다.
</br></br>

## 🗂️ 파일 구조
```
main.cpp
│
├── utils.h
│   └── utils.cpp
│
└── matmul.h
    ├── matmul.cu
    ├── ver3.cu
    ├── ver4.cu
    └── ver5.cu
```
</br>

## ⚙️ 개발 환경
- `CUDA 12.6`
</br>

## 🛠️ 설치 및 실행 방법
- 빌드 명령어 : `make all`
- 실행 방법 &nbsp; &nbsp;: `./main -v 3 -m 320 -k 640 -n 320`
</br>

## 📌 구현 단계
#### ~~ver1~~
- ~~3차원 loop을 이용한 naive한 구현~~
#### ~~ver2~~
- ~~memory coalescing~~
#### ver3
- 공유 메모리
- memory coalescing
#### ver4
- 1D tiling
#### ver5
- 2D tiling
</br>

## ✍️ 구현 원리
<a href="https://tarry-devourer-382.notion.site/SGEMM-1172102a5e3980cfb3f2fcfdf82f6155?pvs=4">이 링크</a>에 업데이트 중입니다.
