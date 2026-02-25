## YOLOv11m Graph-Partitioned Hybrid Inference on Jetson Orin NX

**(DLA INT8 PTQ + GPU FP16, Two-Engine Pipeline)**

YOLOv11m을 대상으로 Jetson Orin NX 16GB에서 **그래프 분할 기반 DLA(INT8)/GPU(FP16) 하이브리드 추론(2엔진 파이프라인)**을 구현했습니다. 하이브리드 INT8에서 발생한 **mAP 붕괴(PTQ collapse)**를 샘플링/재학습(QAT) 대신 **calibration-only padding 분포 변화**로 복구했습니다. 또한 throughput/latency 모드에서 **FPS·지연(p50/p95/p99)·전력·열·프레임당 에너지(J/frame)**를 실측 비교할 수 있도록 벤치마크 파이프라인을 구축했습니다. 

### 대표 결과 (Top Metrics)

* **최대 처리량(throughput):** 60.02 → **91.76 FPS (+52.9%)** 
* **프레임당 에너지(Idle-subtracted):** 0.2095 → **0.1394 J/frame (-33.4%)** 
* **정확도(Track A, mAP50-95):** const PTQ **42.45~43.23 →** mean/random PTQ **48.10~48.71** (FP16 49.52 대비 -0.8~-1.4p 수준)

[Full technical report](./YOLOv11m_Hybrid_Inference_on_Jetson_Orin_NX.md)
<br>

## Problem

* 엣지 환경에서는 **전력·열 예산** 때문에 GPU-only로 모든 추론을 처리하면 운영점이 제한됨
* Orin NX의 DLA는 **CNN 연산에 효율적**이나 attention/transformer 계열 연산은 지원하지 않음 
* 최근 CV 모델은 CNN backbone 외에 attention/transformer 등 **비-CNN 연산**을 포함하는 경우가 많음
* 따라서 모델 전체를 DLA로 옮기기 어렵고, GPU-only는 **에너지 효율 및 지속 성능(thermal)** 측면에서 불리함

## Approach

* **CNN-heavy 구간은 DLA로 offload**, 비-CNN 구간은 GPU에 남기는 방식으로 **파이프라인 KPI 개선**을 목표로 함 
<br>

## Project Info
* **형태:** 개인 프로젝트 (업무 외 시간에 self-directed로 진행)
* **기간:** 2026.01 ~ 2026.02 (6주)
* **역할:** end-to-end

## Tech Stack

* **HW/Platform:** NVIDIA Jetson Orin NX 16GB, DLA, GPU
* **Runtime/Compiler:** TensorRT (INT8 PTQ / FP16), ONNX
* **Languages:** C++, Python, CUDA
* **Profiling/Logging:** CUDA Events, tegrastats(전력/열), (선택) jtop(util)
* **Data/Analysis:** NumPy, Pandas, CSV/JSONL 기반 실험 로그/결과 관리
* **Build/Tools:** CLI 기반 엔진 빌드/벤치 실행 스크립트, Git
<br>

## Contributions

### 1) Split boundary 결정 (단일 경계 유지)

* **경계 선택:** CNN 계열을 최대한 포함하도록 **P3/P4/P5 각 경로에서 C3k2 이후를 단일 경계로 고정** 
* **선정 이유:** 이후 등장하는 **C2PSA는 DLA 미지원(MatMul 계열)**이라 더 깊게 offload하려면 다중 경계(DLA→GPU→DLA)가 필요해지고, 파이프라인 복잡도 및 재양자화/수치 안정성 리스크가 커 **단일 경계(2엔진 파이프라인)**로 범위를 통제

[Technical note (System Overview: Hybrid Split Execution)](https://github.com/enig14/hybrid-inference-experiments/blob/main/docs/YOLOv11m_Hybrid_Inference_on_Jetson_Orin_NX.md#2-system-overview-hybrid-split-execution)

### 2) PTQ mAP 붕괴 복구 (calibration-only padding)

* **현상:** const padding 기반 1K calibration PTQ에서 mAP50-95가 **42.45~43.23** 수준으로 붕괴 
* **원인 파악:** 샘플링 자체보다 **letterbox padding이 유발하는 histogram spike/comb**가 PTQ 안정성을 훼손할 수 있음을 분리 관측 
* **해결:** 운영 전처리는 유지하고 **calibration-only로 mean/random padding**을 적용해 mAP를 **48.10~48.71** 수준으로 안정화(패딩 분포 설계로 복구)
  
[Technical note (PTQ root-cause)](https://github.com/enig14/hybrid-inference-experiments/blob/main/docs/YOLOv11m_Hybrid_Inference_on_Jetson_Orin_NX.md#52-padding-%EB%B6%84%ED%8F%AC%EA%B0%80-%EC%9C%A0%EB%8F%84%ED%95%9C-calibration-cachestep-%EB%B3%80%ED%99%94-%EC%B6%94%EC%A0%81)

### 3) 2엔진 파이프라인 구현 (DLA↔GPU)

* **동기화/연결:** `cudaEventRecord` + `cudaStreamWaitEvent`로 DLA 완료 이후 GPU 실행을 연결(필요 구간만 동기화) 
* **파이프라이닝:** **ring-buffer(NBUF=3)**로 H2D/DLA/GPU/D2H를 슬롯 단위 오버랩하여 throughput 모드 구성 
* **운영 모드:** latency / throughput 모드 분리 운용(측정 목적에 맞게 스케줄링)

### 4) 벤치마크/계측 파이프라인 구축

* **FPS/Latency:** CUDA events 기반 stage/프레임 실행 시간 계측 → FPS 및 **p50/p95/p99 latency**를 모드별로 비교 
* **Energy:** tegrastats 전력 로그를 프레임 타임윈도우에 정렬 → $P_{excess} = P_{avg} - P_{idle}, E_{frame} = P_{excess} / FPS$로 idle-subtracted J/frame 산출 → **GPU-only vs DLA+GPU** 비교 