# 🖼️ 영상처리 실체 (Image Processing) 실습 저장소

충북대학교 OOOO년 O학기 '영상처리 실체' 과목을 수강하며 진행한 모든 실습 코드와 결과물을 아카이빙하는 저장소입니다. 각 주차별(또는 주제별)로 디렉터리를 나누어 실습 내용을 체계적으로 정리했습니다.

## 📚 과목 정보 (Course Information)

- **과목명**: 영상처리 실체 (Core Concepts of Image Processing)
- **수강 학기**: 2025년 1학기
- **주요 사용 언어**: `Python`
- **핵심 라이브러리**: `OpenCV`, `NumPy`, `Matplotlib`

---

## 📂 실습 내용 (Lab Contents)

각 디렉터리는 하나의 영상처리 주제를 다루고 있습니다. 제목을 클릭하면 해당 실습 폴더로 이동합니다.

### 1. [Week01_Image-Basics](<./Week01_Image-Basics>)
- **설명**: 이미지 파일을 읽고 쓰는 방법과 기본적인 픽셀 연산을 다룹니다. 이미지의 색상 공간(RGB, Grayscale)을 이해하고, 특정 영역(ROI)을 잘라내거나 히스토그램을 분석하는 기초를 학습했습니다.
- **주요 개념**: `이미지 입출력`, `Grayscale 변환`, `픽셀 접근`, `ROI(Region of Interest)`, `히스토그램`

### 2. [Week02_Image-Filtering](<./Week02_Image-Filtering>)
- **설명**: 이미지의 노이즈를 제거하거나 특정 효과를 주기 위한 필터링 기법을 실습합니다. 평균값 필터, 가우시안 블러, 미디언 블러 등 다양한 필터의 원리와 적용 결과를 비교 분석했습니다.
- **주요 개념**: `컨볼루션(Convolution)`, `블러링(Blurring)`, `샤프닝(Sharpening)`, `노이즈 제거`

### 3. [Week03_Edge-Detection](<./Week03_Edge-Detection>)
- **설명**: 이미지에서 객체의 외곽선(Edge)을 검출하는 다양한 알고리즘을 구현했습니다. 소벨(Sobel), 라플라시안(Laplacian), 캐니(Canny) 엣지 검출기의 차이점을 이해하고 성능을 비교했습니다.
- **주요 개념**: `미분 필터`, `Sobel`, `Laplacian`, `Canny Edge Detection`

### 4. [Week04_Geometric-Transforms](<./Week04_Geometric-Transforms>)
- **설명**: 이미지의 크기를 조절하거나, 회전시키고, 원근을 바꾸는 등 기하학적 변환을 실습합니다. 아핀(Affine) 변환과 원근(Perspective) 변환의 원리를 이해하고 행렬 연산을 통해 이미지를 변형시켰습니다.
- **주요 개념**: `크기 조절(Scaling)`, `회전(Rotation)`, `이동(Translation)`, `Affine Transform`, `Perspective Transform`

### 5. [Week05_Morphology](<./Week05_Morphology>)
- **설명**: 이미지의 형태를 변형시켜 노이즈를 제거하거나 특정 형태를 부각시키는 모폴로지 연산을 학습합니다. 주로 이진화된 이미지에 적용하여 객체를 분리하거나 합치는 데 사용했습니다.
- **주요 개념**: `침식(Erosion)`, `팽창(Dilation)`, `열림(Opening)`, `닫힘(Closing)`

### 6. [Week06_Image-Segmentation](<./Week06_Image-Segmentation>)
- **설명**: 이미지를 의미 있는 여러 영역으로 분할하는 세그멘테이션 기법을 다룹니다. 특정 임계값을 기준으로 이미지를 흑백으로 나누는 이진화(Thresholding)와 객체의 윤곽선을 찾는 방법을 실습했습니다.
- **주요 개념**: `이진화(Thresholding)`, `적응형 이진화`, `Contours 찾기`

### 7. [Week07_Feature-Detection](<./Week07_Feature-Detection>)
- **설명**: 이미지의 특징점(Feature)을 검출하고, 이를 이용해 객체를 추적하거나 이미지를 매칭하는 방법을 학습합니다. Harris Corner, SIFT, ORB 등의 특징점 검출 알고리즘을 사용했습니다.
- **주요 개념**: `Harris Corner`, `SIFT(Scale-Invariant Feature Transform)`, `ORB(Oriented FAST and Rotated BRIEF)`

---

## 🚀 실행 방법 (How to Run)

각 실습은 독립적인 스크립트로 구성되어 있습니다.

1.  전체 프로젝트에 필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
    > **`requirements.txt` 예시:**
    > ```txt
    > opencv-python
    > numpy
    > matplotlib
    > ```

2.  원하는 실습의 디렉터리로 이동합니다.
    ```bash
    cd <실습_폴더명>  # 예: cd Week02_Image-Filtering
    ```

3.  폴더 내의 파이썬 스크립트(`*.py`)를 실행합니다. 각 폴더의 세부 설명은 해당 폴더 내의 `README.md`(만약 있다면)를 참고해주세요.

## 📝 정리 및 후기

'영상처리 실체' 과목을 통해 이미지 데이터의 가장 낮은 수준인 픽셀부터 시작하여 필터링, 변환, 분석에 이르기까지 영상처리의 핵심적인 파이프라인을 직접 구현해볼 수 있었습니다. 이론으로만 접했던 기술들을 `OpenCV` 라이브러리를 통해 시각적인 결과물로 확인하는 과정이 매우 흥미로웠습니다.

---
*Made by **전양호***
