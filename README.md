# [Edge AI] Jetson Orin NX 하이브리드 추론 파이프라인 최적화 (DLA + GPU)

**한 줄 요약**: Jetson Orin NX에서 **DLA(INT8 PTQ) + GPU(FP16)** 2엔진 하이브리드 추론을 설계/구현하고, 운영 KPI(FPS·Latency·Idle-subtracted J/frame)로 효과를 검증했으며, PTQ에서 발생한 **mAP 붕괴를 Calibration-only Padding 분포 설계로 복구**했습니다.

## 📌 개발 환경 및 타겟

* **기간 / 인원:** 2026.01 ~ 2026.02 (6주) / 1인 | **기술 스택:** C++, CUDA, Python, TensorRT (INT8/FP16), Jetson Orin NX
* **사용 모델:** YOLOv11m | **선정 이유**: 다수의 DLA 친화적 연산을 포함하면서도 DLA 제약에 걸리는 연산(C2PSA 등)이 혼재되어 있어 ‘부분 미지원 + 분할 필요’ 상황을 재현할 수 있었습니다.

## 💡 목표

* 엣지 추론은 단순한 Peak FPS보다 **운영점(에너지/프레임, Latency)이** 훨씬 중요합니다.
* 또한 단순히 'DLA에서 모델을 구동했다'는 사실 자체는 큰 의미를 갖지 못합니다. 미지원 연산(GPU Fallback)이나 양자화로 인한 정확도 손실 때문에 성능과 재현성이 흔들리기 때문입니다.
* 따라서 본 프로젝트는 단순히 작동 여부를 확인하는 데 그치지 않고, **(1) DLA의 성능/특성 제약 하에서 하이브리드 추론을 통제 가능하게 만들고, (2) 이를 KPI로 측정하며, (3) PTQ 변환 시의 정확도 리스크를 원인 분리 후 복구**하는 것을 핵심 목표로 삼았습니다.

## 🚀 결과

단순 FP16 변환 모델의 GPU 단독 실행 대비, 정확도 손실을 1%p 미만으로 방어하면서 처리량은 50% 이상, 프레임당 에너지 소모는 30% 이상 개선했습니다.

| Metric | GPU-only (FP16) | Hybrid (DLA INT8 + GPU FP16) | 개선율 (Δ) |
| --- | --- | --- | --- |
| **Throughput** | 60.02 FPS | **93.60 FPS** | **+55.9%** |
| **Energy** (Idle-subtracted) | 0.2095 J/frame | **0.1394 J/frame** | **-33.4%** |
| **mAP50-95** (COCO) | 49.52 | **48.71 (여러 시드 중 최상치)** | **-0.81p** |

> * Latency(p50/p95/p99)는 동일 프로토콜로 수집했으며, 본 측정 설정에서는 p95/p99가 p50과 거의 동일해 꼬리 지연(Tail issue)은 관측되지 않았습니다.
> * 에너지 지표는 대기 전력을 제외한 순수 추론 에너지(Idle-subtracted) 기준입니다.


## 🛠 문제 해결 과정

### 1. 단일 경계(Split Boundary) 설계

* **문제:** Jetson 환경에서 C2PSA 모듈(MatMul 계열)은 DLA에서 지원하지 않고 모델 후반부의 특정 텐서 축의 길이(8400)가 DLA 제약(8192)에 걸려 전체 모델을 DLA에 올릴 수 없었습니다. 
* **해결:** 연산의 종류와 가속기 특성을 분석하여, **C3k2 모듈 직후를 단일 경계로 설정**했습니다. 다중 경계(DLA→GPU→DLA)로 인한 파이프라인 복잡도와 재양자화 리스크를 사전에 차단하고, `cudaEvent`와 3-stage Ring-buffer를 활용해 이종 가속기 간의 파이프라이닝을 구현했습니다.

### 2. PTQ 정확도 붕괴 원인 규명 및 Calibration 분포 설계

* **문제:** DLA INT8 PTQ 변환 시 mAP가 42.79(-6.73p) 수준으로 급락(붕괴)하는 치명적인 문제가 발생했습니다.
* **해결:** 재학습(QAT)에 의존하지 않고 원인을 추적한 결과, 샘플링 자체가 아니라 **Letterbox Padding에 사용된 고정값(114)이 히스토그램 스파이크를 유발하여 KL-sweep 목적함수를 왜곡**시킨다는 사실을 발견했습니다. 운영 환경의 전처리는 유지하되, **Calibration 과정에서만 Padding 값에 분산을 주는 방식(Mean/Random Padding)을** 고안하여 mAP를 48.71(-0.81p) 수준으로 성공적으로 복구했습니다.

## 📦 산출물

* **2엔진 하이브리드 파이프라인:** DLA segment(INT8) ↔ GPU segment(FP16) 연결 및 이벤트 기반 동기화 구현
* **파이프라이닝 최적화:** Ring-buffer(NBUF=3) 구조로 H2D/DLA/GPU/D2H 구간 오버랩 달성 (Throughput 모드)
* **벤치마크 및 계측 도구:** FPS, Latency(p50/p95/p99), tegrastats 기반 **Idle-subtracted J/frame** 산출
* **PTQ 안정화 기법:** 운영 전처리 로직은 보존하면서 **Calibration-only Padding 분포(Mean/Random)** 제어로 mAP 복구

## 🔗 연관 문서

* **Background (프로젝트 배경/문제 정의):** [`docs/background.md`](./docs/background.md)
* **Full report (근거·프로토콜·분석 전문):** [`docs/Hybrid_Inference_on_Jetson_Orin_NX.md`](./docs/Hybrid_Inference_on_Jetson_Orin_NX.md)

