# Hybrid Inference on Jetson Orin NX (DLA INT8 + GPU FP16, model: YOLOv11m)

## 0. 핵심 요약

### 0.1 정확도 최소 허용 기준 및 평가 결과 (Track A)

> **평가 목표:** 무거운 `m` 모델을 이종 가속기로 분할 실행할 때 발생하는 연산 및 메모리 비용 증가를 정당화하기 위해, 정성적 체감이 아닌 정량적 **'최소 허용 기준'을** 정의했습니다.
> **Pass/Fail 기준:** `m` 모델과 하위 `s` 모델 간의 원래 성능 간극(약 4.6p) 중 **최소 50% 이상을 보존(목표 하한선: 47.22 mAP)해야만** 시스템 효율 측면에서 하이브리드 운용의 타당성이 확보된다고 판단했습니다.
> > * **최소 허용선 산출식:** $mAP_{floor} = mAP_{s} + 0.5 \times (mAP_{m} - mAP_{s})$
> * $47.22 = 44.92 + 0.5 \times (49.52 - 44.92)$
> * **평가 조건:** 모든 mAP는 동일한 Track A 설정(`coco val2017`, `imgsz=640`, `conf=0.001`, `iou=0.45`, `max_det=1000`)에서 평가되었습니다.
> 
| 모델 (체급) | 실행 엔진 (정밀도) | Calibration 패딩 정책 | 평균 mAP50-95 (Seed: 42/84/126) | 허용 기준 판정 (Pass/Fail) |
| --- | --- | --- | --- | --- |
| **YOLOv11s** | GPU-only (FP16) | - (Lower-bound) | 44.92 | - |
| **YOLOv11m** | GPU-only (FP16) | - (Target) | 49.52 | - |
| **YOLOv11m** | **Hybrid (INT8+GPU)** | Const (고정 단색) | 42.79 | **Fail** (s 모델보다 성능 하락) |
| **YOLOv11m** | **Hybrid (INT8+GPU)** | **Mean (이미지 평균색)** | 48.44, 48.12, **48.62** | **Pass** (허용선 대비 평균 +1.17p) |
| **YOLOv11m** | **Hybrid (INT8+GPU)** | **Random (채널별 난수 단색)** | 48.15, 48.10, **48.71** | **Pass** (허용선 대비 평균 +1.10p) |

*요약: 1K PTQ 변환 시 고정색(Const) 패딩은 `s` 모델보다 못한 성능으로 붕괴(Fail)했으나, Mean/Random 패딩 분산을 통해 최소 허용 기준을 여유 있게 통과(Pass)하며 `m` 모델 체급의 효용성을 성공적으로 방어했습니다.*


### 0.2 하이브리드 성능 및 전력 효율 요약 (Track B)

> **요약:** 하이브리드 추론 적용 시, GPU 단독 실행 대비 **Throughput(FPS)은 52.9% 향상**되고 **프레임당 에너지 소모(E_frame)는 33.4% 절감**되었습니다. 파이프라이닝 및 가속기 특성으로 인해 지연 시간(Latency)은 13% 내외로 소폭 증가했습니다.

| 평가 지표 (Metric) | 측정 모드 | GPU-only (FP16) | Hybrid (DLA INT8 + GPU) | 증감률 (Δ) |
| --- | --- | --- | --- | --- |
| **Throughput** (FPS) | 캡 해제 (최대 성능) | 60.02 FPS | **91.76 FPS** | **+52.9%** 🚀 |
| **에너지 효율** (E_frame) | 캡 해제 (Idle-subtracted) | 0.2095 J | **0.1394 J** | **-33.4%** 🔋 |
| **지연 시간** (Latency p50) | Latency 최우선 모드 | 15.10 ms | **17.04 ms** | +12.91% ⏱️ |

* 0.2 용어 정리
- **const padding:** 운영 전처리에서 사용하는 기본 letterbox padding 방식(고정 단색(114,114,114)).
- **mean padding:** *이미지별* 입력 이미지의 RGB 평균 단색을 padding 영역 전체에 적용하는 방식
- **random padding (이미지별 단색)**  
  - 정의: *이미지마다* padding 영역 전체에 적용할 RGB 단색(solid color)을 난수로 샘플링  
  - 분포: 채널별 `N(μ_c, 2σ_c)` (train-set 통계 기반)에서 샘플링 후 `[0,255]` clip  
  - ⚠️ 주: 운영/추론 전처리는 `const(114)` 유지(=calibration-only 적용)  
  - 의도(원인 분리): mean padding의 효과가 “mean 값 자체”인지 “값 분산(흩어짐)”인지 분리하기 위한 ablation  
  - 구현: 학습이미지 전체에 대해서 채널별 통계 수집 → 이미지별 RGB 단색 생성 → padding 영역 전체에 적용

#### 0.3 측정 프로토콜 및 실험 조건 세팅

* **전처리(Preprocess):** `imgsz=640`, `batch=1`, RGB NCHW pack, Letterbox (Center align 후 Padding 적용), 0~1 정규화
* **Calibration-only 증명:** 운영/추론 전처리는 `const(114)`로 엄격히 유지되었으며, 패딩 정책 변경은 Calibration용 NPY 파일 생성 단계에서만 격리되어 수행되었습니다.
* *Calib 생성 커맨드:* `python make_calib_npy.py --pad_mode mean --out calib_mean.npy`
* *Bench 실행 커맨드:* `./bench --mode=hybrid ...` (별도의 패딩 옵션 없이 기본 전처리 수행)


* **에너지 지표 산출 근거 (E_frame):**
* `tegrastats`의 `VDD_IN` 전력망 데이터를 기준(통계 스크립트 기반)으로 산출했습니다.
* $P_{idle}$: 벤치마크 시작 전 시스템 안정화(30초) 구간의 평균 대기 전력
* $P_{avg}$: Warm-up(초기 10프레임) 제외 후 측정 구간의 평균 전력
* $P_{excess} = P_{avg} - P_{idle}$
* **[계측 요약]** GPU-only (`12.58W / 60.02 FPS` $\approx 0.209J$), Hybrid (`12.79W / 91.76 FPS` $\approx 0.139J$)


* **Latency 비교 지표 정의:** * 파이프라인 대기(Queueing)가 포함되는 Throughput 모드의 지연 시간은 시스템 오버헤드를 포함하므로, **순수 처리 속도 비교는 'Latency 최우선 직렬 모드'의 데이터(p50)만을 기준**으로 삼았습니다.

## 1. Project Environment & Scope (환경 및 범위)

### 1.1 하드웨어 및 소프트웨어 스택
- Device: Jetson Orin NX 16GB
- C++, CUDA, Python, TensorRT (INT8/FP16) 등
* **Power mode**: `nvpmodel = MAXN`

* **jetson_clocks**: requires root (`sudo jetson_clocks`)

* **SoC / Platform**: tegra234 (Jetson Orin NX)

* **Kernel**: Linux 5.15.148-tegra (build: 2025-09-18)

* **Jetson Linux (L4T)**: 36.4.7-20250918154033 (from `nvidia-l4t-cuda`)

* **CUDA**: 12.6 (e.g., `cuda-cudart-12-6 12.6.68-1`)

* **cuDNN**: 9.3.0.75 (`libcudnn9-cuda-12 9.3.0.75-1`)

* **TensorRT**: 10.3.0.30 (`libnvinfer10 10.3.0.30-1+cuda12.5`)

* **VPI**: 3.2.4 (`libnvvpi3 3.2.4`)

* **Python**: 3.10.12

### 1.2 타겟 모델 및 평가 범위
* **모델:** YOLOv11m (COCO pretrained)
* **본 문서 범위:** Jetson Orin NX 16GB에서 YOLOv11m을 **DLA(INT8 PTQ) + GPU(FP16)로** 분할 실행하기 위한 (i) 그래프 분할/엔진 빌드, (ii) 2-엔진 실행 파이프라인 및 스케줄링, (iii) 성능·지연·에너지 지표 측정 프로토콜, (iv) PTQ 정확도 붕괴의 원인(캘리브레이션 입력 분포/letterbox padding) 규명 및 복구 근거를 다룹니다.
  - **포함:** calibration-only 입력 분포 설계(const vs mean/random padding), cache(step/scale) 변화 및 KL 기반 threshold 이동 분석, mAP·FPS·Latency(p50/p99)·Idle-subtracted J/frame 비교
* **본 문서 범위 외**
  - 학습 중심 최적화(QAT/재학습으로 SOTA 달성)
  - 모델 전체를 단일 가속기에서 100% 실행
  - 모든 플랫폼/모델로의 일반화

## 2. Architecture: 하이브리드 파이프라인 설계 (DLA + GPU)

### 2.1 End-to-end pipeline

```text
Input Image
   |
   v
[Preprocess]
- resize/letterbox, normalize, layout convert
   |
   v
+--------------------------------------------------+
|   DLA segment (INT8, PTQ)                        |
|  - Backbone (shared trunk)                       |
|  - Per-scale branches up to C3k2                 |
|      P3 -> C3k2 -> (cut after C3k2)              |
|      P4 -> C3k2 -> (cut after C3k2)              |
|      P5 -> C3k2 -> (cut after C3k2)              |
+--------------------------------------------------+
   |            |             |
   | P3 feat    | P4 feat     | P5 feat
   v            v             v
+--------------------------------------------------+
|   GPU segment (FP16)                             |
|  - Modules after the split                       |
|  - Includes C2PSA and downstream dependent layers|
|  - Detect head / post-feature fusion             |
+--------------------------------------------------+
   |
   v
[Postprocess]
- decode, NMS, format conversion
   |
   v
Detections
```
### 2.2 분할(Split) 경계 설정 기준 및 정당화

* YOLOv11m의 C2PSA 모듈(MatMul 연산)은 DLA에서 지원하지 않습니다.
* DLA 커버리지를 극대화하기 위해 P3/P4/P5 경로의 **C3k2 모듈 직후**를 경계로 설정하고, 이후를 GPU로 넘기는 2엔진(Split) 파이프라인을 설계했습니다.
* **FP16 Export 근거:** DLA 출력을 INT8이 아닌 FP16으로 Export 한 이유는, GPU가 INT8 입력을 받을 때 발생하는 Re-quantization 오버헤드와 정밀도 변환 리스크를 제거하기 위함입니다.

### 2.3 이종 가속기 I/O 매핑 및 메모리 직접 연결 (Device-to-device)

DLA 출력 버퍼(`dOutP3/P4/P5`)를 GPU에서 접근 가능한 Device memory로 유지하여, Host 메모리(D2H/H2D)를 경유하지 않는 다이렉트 연결을 구현했습니다. TRT 엔진 재빌드 시 Binding Index가 변경될 리스크를 차단하기 위해 **명시적인 이름 기반 바인딩**을 적용했습니다.

* **DLA Outputs:** `P3_cut`, `P4_cut`, `P5_cut`
* **GPU Inputs:** `P3_in`, `P4_in`, `P5_in`
* *실행 인자 예시:* `--connect "P3_cut:P3_in, P4_cut:P4_in, P5_cut:P5_in"`

### 2.4 파이프라이닝 및 이벤트 동기화: Ring-buffer (NBUF=3) 설계

Throughput 모드에서는 `NBUF=3` 슬롯을 사용하여 H2D -> DLA -> GPU -> D2H 과정을 오버랩합니다. DLA가 작업 중인 버퍼를 GPU가 덮어쓰는 경합(Data Race)을 막기 위해, DLA 스트림에 `cudaEventRecord(evDLA_done)`를 기록하고 GPU 스트림에서 `cudaStreamWaitEvent`로 의존성을 대기하는 효율적인 하드웨어 큐잉을 구현했습니다.


## 3. Performance & Efficiency: 시스템 지표 분석

본 장에서는 GPU-only(FP16) 환경과 Hybrid(DLA INT8 + GPU FP16) 환경에서의 시스템 지표(FPS, Latency, Energy)를 비교합니다.

### 3.1 벤치마크 결과 요약 (Throughput vs Latency 모드)

**[표 3-1] FPS 캡 해제 / 최대 Throughput 비교 (Baseline = GPU-only throughput)**

| Mode | Engine | FPS | FPS 증감 | Lat p50 (ms) | Lat 증감 | E_frame (J) | Energy 절감 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU-only (baseline) | YOLOv11m FP16 | 60.02 | - | 14.69 | - | 0.2095 | - |
| Hybrid (throughput) | INT8(DLA) + FP16(GPU) | 93.60 | **+55.95%** | 18.40 | +25.3% | 0.1394 | **-33.4%** |

*(※ p99 Latency는 GPU-only 16.61ms, Hybrid 19.58ms로 p50과 큰 차이가 없어 꼬리 지연(Tail issue)은 관측되지 않았습니다.)*

### 3.2 Throughput 상승의 원인과 Latency 증가의 Trade-off

**[표 3-2] 하이브리드 추론 시간 상세 분해 (p50 기준)**

| 단계 (Stage) | 시간 (ms) | 점유율 (%) | 비고 |
| --- | --- | --- | --- |
| Pre-task | 0.70 | 4.1% | Letterbox + Pack |
| Data Move | 0.22 | 1.3% | Host to Device |
| **DLA Segment** | **9.11** | **53.5%** | **Backbone (INT8)** |
| **GPU Segment** | **6.87** | **40.3%** | **C2PSA+Heads (FP16)** |
| Data Move | 0.12 | 0.7% | Device to Host |
| **Total (e2e)** | **17.04** | **100%** | Sync 오버헤드: 미미|

Hybrid 모드에서 Throughput은 상승했지만 개별 프레임의 Latency가 증가하는 현상은 다음 두 가지 컴퓨터 시스템 이론으로 설명할 수 있습니다.

1. **Throughput 상승 (파이프라이닝):** `NBUF=3` 링 버퍼를 통해 DLA와 GPU가 서로 다른 프레임을 병렬 처리합니다. 전체 성능은 병목 구간에 의해 결정되며 **(암달의 법칙)**, 본 실험에서는 DLA(9.11ms)가 병목입니다. 따라서 이론적 최대 FPS는 $1000 / 9.11 \approx 110$ FPS이며, 측정된 93.60 FPS는 전처리와 후처리 오버헤드를 제외하면 이론치에 근접한 수준으로 정상 작동함을 의미합니다.
2. **Latency 증가의 원인 (리틀의 법칙):** 리틀의 법칙($L = \lambda W$)에 따라 시스템 내 체류 프레임 수($L=3$, 버퍼)를 꽉 채워 Throughput($\lambda$)을 높이면 체류 시간($W$, Latency)은 길어집니다. DLA(9.11ms)와 GPU(6.87ms)의 처리 속도가 비대칭적이므로, 빠른 GPU 구간 앞단에서 필연적으로 **큐잉 대기 시간(Queueing Wait Time)**이 발생하여 개별 프레임의 Latency가 직렬 실행(GPU-only) 때보다 무조건 늘어날 수밖에 없는 수학적/구조적 결과를 가집니다.

### 3.3 전력 효율(Idle-subtracted J/frame) 개선 효과 및 해석

* **절감의 핵심:** DLA는 GPU 대비 Watt당 성능이 높은 ASIC입니다. 모델 연산량의 대부분을 차지하는 무거운 Backbone 연산을 DLA로 이관함으로써 GPU 사용률(Load)을 크게 낮춘 것이 전력 감소(-33.4%)의 주효한 원인입니다.
* **실무적 이점:** 전력 소모의 감소는 곧 발열 감소로 이어집니다. 쿨링 팬이 없는(Fanless) 오토모티브 기기나 고온 환경에서 **열 스로틀링(Thermal Throttling) 발생 시점을 늦추는 실질적인 엣지 성능 향상**으로 이어집니다.

---

## 4. Troubleshooting: PTQ 정확도 붕괴 원인 분석 및 복구

### 4.1 문제 발생: 하이브리드 모델의 mAP 급락

파이프라인 구축 후 DLA 구간을 INT8 PTQ로 양자화했을 때, mAP50-95가 원본 49.52에서 **42.79로 심각하게 붕괴(Fail)** 하는 현상이 발생했습니다. 초기에는 calibration 이미지의 대표성 문제로 판단하여 stratified 등 샘플링 방식을 변경해 보았고, 일부 조건에서는 정확도가 부분적으로 복구되기도 했습니다. 그러나 “왜 특정 샘플링에서만 회복되는지”를 설명할 수 있는 인과관계가 부족하여, 샘플링 외 요인을 통제하는 방식으로 재분석에 착수했습니다.

---

### 4.2 현상 관측: 요약 통계는 유사하나 Calibration step이 점프

샘플링 외 변인을 통제하며 추적한 결과, 전처리 단계의 **Letterbox Padding(고정 단색값 사용)** 정책이 TensorRT PTQ의 스케일 추정 결과에 지배적인 영향을 미친다는 점을 확인했습니다.

패딩 정책(고정값 vs 분산)에 따라 calibration cache(step)가 어떻게 변하는지 확인하기 위해, step 변화율이 가장 큰 Top-3 텐서를 추적했습니다. 그 결과, **요약 통계(mean/std/q99/amax)는 거의 동일함에도 불구하고**, padding 정책에 따라 **step 값이 1.5배~2배까지 크게 달라지는 현상**을 관측했습니다.

**[표 4-1] cache(step) 이동: Top-3 tensors (activation 제외)**

> ⚠️ 실제로는 아래에 표시된 레이어 직후의 activation(SiLU) 출력 텐서에서 편향(분포 변화)이 가장 크게 관측되었습니다.
> 다만 activation은 비선형 함수(SiLU)이므로 입력 분포 변화가 비선형적으로 재매핑되어, const vs mean/random에 의해 유도된 원인을 직접적으로 해석·역추적하기 어렵습니다.
> 따라서 본 절에서는 “activation에서 새로 생성된 현상”이라기보다, **이전 선형 구간(Conv)에서 이미 형성된 분포 차이가 SiLU의 비선형성에 의해 증폭되어 관측된 것**으로 보고, 선형 레이어 출력 텐서를 case study 대상으로 선정했습니다.

| Tensor Name                               | const (42,84,126)                  | mean (42,84,126)                   | random (42,84,126)                 | ratio mean/const (42,84,126) | ratio random/const (42,84,126) | 비고        |
| ----------------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------- | ------------------------------ | --------- |
| /model.2/m.0/cv1/conv/Conv_output_0       | 0.05063288, 0.05078436, 0.05107691 | 0.06261881, 0.07677290, 0.06285256 | 0.08151972, 0.08970439, 0.08915153 | 1.23, 1.51, 1.23             | 1.45, 1.77, 1.75               | spike에 민감 |
| /model.2/m.0/cv2/conv/Conv_output_0       | 0.09571205, 0.09488328, 0.10304550 | 0.12519260, 0.14536300, 0.12153770 | 0.17240120, 0.15564600, 0.17214800 | 1.31, 1.53, 1.18             | 1.80, 1.64, 1.67               |           |
| /model.2/m.0/m/m.0/cv2/conv/Conv_output_0 | 0.03263176, 0.03186469, 0.02753039 | 0.04738905, 0.04743533, 0.04740598 | 0.05985778, 0.06087017, 0.05779142 | 1.45, 1.49, 1.72             | 1.83, 1.91, 2.10               |           |

또한 case study 텐서(`/model.2/m.0/cv1/conv/Conv_output_0`)에 대해 통계 분포의 변화량(Δ%)을 확인한 결과, **step이 크게 달라지는 현상은 tail(amax/q99.99) 변화로 설명되지 않음**을 확인했습니다.

**[표 4-2] 통계 분포 변화 (Δ% = (mean−const)/const): `/model.2/m.0/cv1/conv/Conv_output_0`**

| tensor                              |   Δmean | Δmean% |      Δstd |  Δstd% |     Δq99 |  Δq99% |   Δq99.9 | Δq99.9% | Δq99.99 | Δq99.99% |
| :---------------------------------- | ------: | :----: | --------: | :----: | -------: | :----: | -------: | :-----: | ------: | :------: |
| _model.2_m.0_cv1_conv_Conv_output_0 | 0.00162 |  0.41% | -0.000679 | -0.06% | -0.00254 | -0.09% | -0.00111 |  -0.03% |       0 |  <0.01%  |

> 표기 규칙: Δ = (mean − const), Δ% = 100 × Δ / const

---

### 4.3 가설 수립 및 메커니즘 분석: Histogram spike와 KL 목적함수의 톱니(saw-tooth)화

위 현상(“요약 통계는 유사하나 step이 점프”)은 단순히 분포의 꼬리(tail)가 길어져서 발생한 문제가 아니라, **히스토그램 이산화(binning) 과정에서 특정 bin에 질량이 집중되는 spike/comb 패턴**으로 인해 KL 기반 threshold 선택이 불안정해지는 메커니즘으로 설명할 수 있다고 가정했습니다.

이 가설은 다음의 연쇄 구조로 검증했습니다.

1. **spike 생성 반복성:** const 조건에서 정의된 “스파이크(spike_bins)”가 동일 1K 이미지에서 반복적으로 생성되는지 확인합니다.
2. **전역 톱니 성분(net reduction):** mean/random에서 기존 스파이크가 약화되더라도, 신규 스파이크가 생길 가능성을 고려하여 KL(T)의 전역 톱니 성분이 순감(net reduction)되는지 확인합니다.
3. **최적점 이동:** 톱니 성분 변화가 실제 `best_Thr` 선택의 이동으로 이어지는지 확인합니다.

#### 4.3.1 Stage-1: “const에서 정의한 스파이크”의 반복성 변화

본 절은 “const padding에서 관측된 spike/comb 패턴이 일부 예외 이미지가 아니라 **대부분 이미지에서 반복되는 공통 패턴**인지”를 확인하고, padding 정책 변경(mean/random)이 **그 spike_bins** 의 충족 빈도를 어떻게 바꾸는지 정량화합니다.
※ 여기서는 “모든 스파이크가 사라졌다”를 주장하지 않습니다. mean/random에서 **다른 위치의 신규 스파이크(emergent spikes)** 가 생길 가능성은 열어두며, 그 가능성까지 포함한 최종 판단은 4.3.2에서 KL(T)의 전역 톱니 성분 지표로 수행합니다.

* `spike_bins`: const histogram에서 spike가 큰 bin들을 상위 분위수(예: 99.5%) 기준으로 선택한 bin index 집합
* `spike_hit_rate`(image, ch): feature 원소 중 spike_bins에 떨어지는 비율
* `heavy-hit`: `spike_hit_rate ≥ τ` (τ=0.2, 0.4)

**[표 4-3] spike_hit_rate 분포 요약(Top 채널 2개; spike_bins는 const 기준으로 고정)**

| ch | metric     | const |  mean | random |
| -: | ---------- | ----: | ----: | -----: |
| 26 | P(hit≥0.2) | 0.913 | 0.038 |  0.008 |
| 26 | P(hit≥0.4) | 0.053 | 0.002 |  0.001 |
| 25 | P(hit≥0.2) | 0.917 | 0.353 |  0.007 |
| 25 | P(hit≥0.4) | 0.053 | 0.022 |  0.000 |

채널 단 분포를 레이어 단으로 요약하기 위해, const 기준 스파이크가 큰 Top-K 채널을 선택하여 LayerScore를 계산했습니다(Top-K=5).

* Top-K=5: `ch21, 7, 25, 22, 26`
* `layer_hit_i = mean_c(hit_{i,c})`

**[표 4-4] LayerScore 요약(Top-K=5; spike_bins는 const 기준으로 고정)**

| policy | LayerScore_mean(hit) | P(layer_hit≥0.2) | P(layer_hit≥0.4) |
| ------ | -------------------: | ---------------: | ---------------: |
| const  |             0.370893 |            0.957 |            0.352 |
| mean   |             0.186463 |            0.420 |            0.010 |
| random |             0.113888 |            0.092 |            0.006 |

**결론(4.3.1의 범위):** const에서 정의된 legacy spike condition은 동일 1K 이미지에서 **반복적으로 충족되는 공통 패턴**이며, mean/random padding에서는 그 충족 빈도(반복성)가 크게 감소합니다.
단, 신규 스파이크 가능성까지 포함한 KL 목적함수 전반의 안정화 여부는 다음 절에서 검증합니다.

#### 4.3.2 Stage-2: KL(T) 전역 톱니 성분(net saw-tooth) 정량화 및 best_Thr 이동

4.3.1에서 legacy spike condition이 약화되었더라도, mean/random에서 다른 위치의 emergent spikes가 새로 생길 수 있습니다. 따라서 본 절은 특정 spike의 “존재 여부”가 아니라, KL(T)의 **전역 톱니(saw-tooth) 성분을 정량화**하여 padding 정책 변경이 목적함수의 불안정성을 **순감** 시키는지 확인합니다.

* `TV_norm`: KL(T) 인접 변화량 총변동(정규화) — 클수록 톱니 성분이 큼
* `HFE_norm`: 2차 차분 기반 고주파 성분(정규화) — 클수록 톱니 성분이 큼

**[표 4-5] KL(T) 톱니성분 지표 요약(정책별 평균)**

| policy | TV_norm | HFE_norm |
| ------ | ------: | -------: |
| const  | 89.2806 |  0.07447 |
| mean   | 36.4092 |  0.03169 |
| random | 14.4377 |  0.01142 |

해석: mean/random에서 `TV_norm`, `HFE_norm`이 크게 감소했으며, 이는 신규 스파이크 가능성을 포함한 전역 관점에서도 KL 목적함수의 톱니 성분이 **유의미하게 순감(net reduction)** 되었음을 의미합니다.

> **[그림 4-1] KL(T) curves under different padding policies**
> ![KL-curve](./assets/figures/kl-curve.png)
> 전역 지표(TV_norm/HFE_norm) 감소와 일관되게, mean/random에서 KL(T) 곡선의 톱니 성분이 완화됨을 시각적으로 확인할 수 있습니다.

다음으로, 톱니 성분이 약화된 조건에서 global optimum 선택이 실제로 어떻게 달라지는지 `best_Thr`를 seed별로 비교했습니다.

**[표 4-6] best_Thr 요약(각 seed 별)**

| Padding policy | seed 42 | seed 84 | seed 126 | hit ceil |
| -------------- | ------: | ------: | -------: | -------: |
| const          |   3.969 |   3.969 |    3.969 |    False |
| mean           |   7.938 |   7.938 |    7.938 |    False |
| random         |   7.938 |   7.938 |    7.938 |    False |

> NOTE: `best_Thr`는 KL-sweep에서 선택된 threshold이며, **abs histogram(range=[0,32], bins=2048)에서** 계산했습니다.
> step size는 32/2048 = 0.015625이므로, 3.969 ≈ 254×0.015625, 7.938 ≈ 508×0.015625처럼 bin-grid에 스냅됩니다.

**결론(4.3.2):** mean/random에서는 신규 스파이크 가능성을 포함한 전역 관점에서도 KL(T)의 톱니 성분이 크게 감소했으며(TV_norm/HFE_norm), 이에 동반하여 선택되는 `best_Thr`가 seed 전반에서 일관되게 이동했습니다.

---

### 4.4 문제 해결: Calibration-only Mean/Random Padding 도입

위 메커니즘 분석을 바탕으로, 히스토그램 특정 bin에 질량이 집중되는 spike/comb 패턴을 완화하기 위해 **calibration 이미지 생성 단계에서만** 패딩 값에 분산을 주는 방식을 도입했습니다.

* **Mean padding:** 이미지별 평균 RGB 색상으로 패딩합니다.
* **Random padding:** 채널별 학습 데이터 통계를 $\mathcal{N}(\mu, 2\sigma)$로 두고 난수 단색을 생성하여 패딩합니다.

분산 패딩을 적용한 결과, KL 목적함수의 톱니 성분이 완화되고(best_Thr가 seed에 대해 일관되게 이동), step 값이 안정화되면서 mAP가 최소 허용 기준(47.22)을 넘는 **48.39 / 48.32** 수준으로 복구되었습니다.
단, **실제 운영/추론 환경의 전처리(입력 letterbox)는 기존 const padding을 그대로 유지**하며, 본 수정은 calibration 입력 분포 설계(calibration-only)로 제한됩니다.


## 5. Evaluation Protocol (평가 기준 및 측정 방법론)

### 5.1 성능/전력 계측 프로토콜 (Idle-subtracted 산출 식)

단순 평균 전력이 아닌, **대기 전력(Idle baseline)을 뺀 순수 추론 에너지**를 계산하여 공정성을 확보했습니다.

1. **$P_{idle}$:** 워크로드 실행 전, 시스템 안정화 상태의 측정 전력.
2. **$P_{avg}$:** 벤치마크 실행 구간(warmup 제외) 동안의 평균 전력.
3. **산출:** $P_{excess} = P_{avg} - P_{idle}$, $E_{frame}(J) = P_{excess}(W) / FPS$

### 5.2 정확도 허용 기준 설정 근거

본 프로젝트는 '돌아간다'에 만족하지 않고, 정량적 하한선(47.22 mAP)을 설정했습니다. `m` 모델을 억지로 양자화해서 가벼운 `s` 모델의 FP16 성능 수준으로 떨어진다면, 굳이 무거운 연산을 감당할 이유가 없기 때문입니다. 따라서 `m`과 `s` 모델의 성능 격차 중 최소 50% 이상을 보존해야만 하이브리드 도입의 타당성이 있다고 기준을 세웠습니다.

## 6. Conclusion & Future Work (결론 및 향후 과제)

### 6.1 결론 및 배운 점

* Jetson Orin NX 환경에서 DLA + GPU 하이브리드 실행을 통해 정확도 손실을 최소화(-1%p 내외)하면서 **최대 처리량을 +50% 이상 향상시키고 전력 소모를 30% 절감**했습니다.
* PTQ 양자화 과정에서 발생하는 치명적인 정확도 저하가 단순한 샘플링 문제가 아니라, **고정값 패딩으로 인한 히스토그램 스파이크와 KL-Divergence 목적 함수의 왜곡(톱니화)** 로 설명했습니다. 이를 Mean/Random 패딩으로 분산시켜 성공적으로 복구했습니다.

### 6.2 향후 과제

* **교차 검증 및 일반화 평가:** 본 *calibration-only 입력 분포(패딩) 설계* 방법론을 **타 객체탐지 모델**(YOLO 외) 및 **CNN 백본이 앞단에 배치된 타 태스크 모델**(분류/세그 등)로 확장 적용하여, 동일한 진단 지표(legacy spike 반복성, KL(T) 톱니 성분 지표, best_Thr 이동)가 재현되는지 검증할 예정입니다. 또한 **타 NPU 플랫폼/툴체인**에서도 동일한 현상이 관측되는지 교차 검증하여, 방법론의 플랫폼 독립성을 평가할 계획입니다. (추가로, 고정 패딩 반복 삽입이 통계 추정에 영향을 줄 수 있다는 관점에서 LLM의 PAD 토큰/시퀀스 길이 구성 문제로의 개념적 확장 가능성도 가설로 검토합니다.)

* **후처리 성능 최적화:** 모델 후처리(Decode/NMS) 구간을 대상으로 CUDA 커널 레벨에서 병목을 식별하고, 커스텀 커널/메모리 접근 최적화 등을 통해 end-to-end 지연 및 처리량을 추가 개선할 예정입니다.

# Appendix

## Appendix A. 전체 프로젝트 타임라인(대략 6주)

### Phase 1. Baseline & Bring-up (초기 1주 내외)

* YOLOv11m DLA/GPU 분할 경계 확정, 엔진 빌드/실행 파이프라인 구축
* Hybrid(INT8)에서 **mAP 붕괴** 최초 관측 → “정확도 리스크” 등록

### Phase 2. Throughput/Latency 파이프라인 구축 (1~4주 구간에 걸쳐 병행)

* **NBUF ring-buffer + CUDA event 동기화**로 DLA/GPU 파이프라이닝 구현
* throughput/latency 모드 정리, 안정성(프레임 정합/데드락 방지) 검증
* power 측정 체계도 이 시기에 같이 고도화

### Phase 3. 정확도 복구 전략 탐색 (중반, 여러 번 반복)

* QAT 검토 → **DLA Q/DQ 제약** 확인 후 실질적으로 배제
* PTQ에서 해결하는 방향으로 선회(캘리브레이션 설계/입력 분포에 집중)

### Phase 4. PTQ 변수 스윕 (중~후반 반복)

* 샘플링(랜덤/stratified/tail-mix)과 입력 분포 요인들을 교차 실험
* 그중 **letterbox padding 정책(Const vs Mean/Random)이** mAP에 가장 큰 영향을 주는 축임을 확인

### Phase 5. 최종 벤치마크/리포팅 (후반 1~2주 내외)

* Track A/B 수집, 병목 분석, trade-off(정확도·성능·전력/에너지) 정량화
* 운영 관점에서 “쓸 수 있는 설정”을 결정 가능한 수준으로 패키징

### Phase 6. Root Cause Analysis (마무리 단계, 마지막 1주 정도)

* padding이 유발하는 histogram 특성(spike/bin-mass shift)이
  KL 기반 threshold 선택에 **saw-tooth 불연속성**을 만들고 calibration cache(step/scale)를 왜곡한다는 메커니즘을 정리/증명


## Appendix B. Hybrid 엔진 구성 및 실행(Implementation)

### B.1 엔진 구성

* GPU-only FP16 엔진
* Hybrid

  * DLA INT8 엔진(PTQ): backbone + C3k2 이전까지
  * GPU FP16 엔진: DLA 이후 + head

### B.2 실행 모드/CLI 예시

* GPU-only

  * `./bench --mode=gpu <engine.plan> [your/valid/set/npy] --sched=throughput --pred_json=...`
* Hybrid

  * `./bench --mode=hybrid <dla.plan> <gpu.plan> [your/valid/set/dir] --sched=throughput --pred_json=...`

<br/>

## Appendix C. Reproducibility: End-to-end 재현 절차(1~5)

> 목적: 본 문서의 핵심 결과(PTQ mAP 급락/회복, cache(step) 변화, 하이브리드 성능/전력 지표)를 동일한 실험 순서로 재현할 수 있도록, 최소 단위 커맨드 플로우를 기록합니다.
> 전처리 변경은 운영단계에서는 제외하고, **calibration 입력 생성 단계에서만 padding 정책을 바꿉니다**

### C.0 공통 전제(고정 조건)

* Device: Jetson Orin NX 16GB, `nvpmodel=MAXN`, (가능하면) `sudo jetson_clocks`
* Input: `imgsz=640`, `batch=1`, RGB, NCHW, normalize(0~1)
* Dataset: COCO train2017(칼리브레이션 입력 생성), COCO val2017(평가/벤치)
* 평가 트랙

  * Track A(mAP): `conf=0.001, iou=0.45, max_det=1000`
  * Track B(운영/벤치): `conf=0.25, iou=0.45, max_det=300`

### Step 1) Calibration 입력 생성(.npy) — padding policy sweep

> NOTE: `--out_list`는 이번 실행에서 선택된 이미지 경로를 기록하기 위한 출력 옵션
> `--list`는 입력 이미지 후보를 고정하기 위한 옵션이고, `--num`은 해당 리스트에서 사용 장수(N)를 강제하여 정책 비교(const/mean/random) 시 동일 장수를 재현시켜 줍니다.
> `--seed`는 이미지 입력 순서를 재현합니다.
> 따라서 재현 절차에서는 `--list`를 사용할 때 `--num`을 함께 사용하며(필수), 필요 시 `--seed`로 결정성을 고정합니다.

**1-A. const padding(참조)**

```bash
python3 make_calib_npy.py \
  --image_dir [image/set/path] \
  --seed [seed] \
  --num 1000 \
  --out_list [out/images/list] \
  --out [const_s{seed}.npy] \
  --size 640 \
  --to_rgb \
  --pad_mode const
```

**1-B. mean padding**

```bash
python3 make_calib_npy.py \
  --list [out/images/list] \
  --seed [seed] \
  --num 1000 \
  --out [mean_s{seed}.npy] \
  --size 640 \
  --to_rgb \
  --pad_mode mean
```

**1-C. random padding (단색)**

```bash
python3 make_calib_npy.py \
  --list [out/images/list] \
  --seed [seed] \
  --num 1000 \
  --out [rand_s{seed}.npy] \
  --size 640 \
  --to_rgb \
  --pad_mode rand \
  --pad_stats_json [your_stats.json]
```

> ⚠️ 주: 운영/추론 시 전처리는 **const padding(114) 그대로** 유지하며, mean/random은 **PTQ calibration 입력 생성 단계에만** 적용합니다.

### Step 2) TensorRT 엔진 빌드 — (A) GPU-only FP16, (B) Hybrid용 DLA INT8 + GPU FP16

#### 2-A) GPU-only FP16 엔진

```bash
python3 build_engine_cli.py \
  --onnx_path [YOLOv11m.onnx] \
  --engine_path [YOLOv11m_fp16.plan] \
  --precision fp16
```

#### 2-B) Hybrid: DLA INT8 엔진(PTQ)

```bash
python3 build_engine_cli.py \
  --onnx_path [dla_part.onnx] \
  --calib_npy [const/mean/random_s{seed}.npy] \
  --engine_path [dla_int8_{policy}_s{seed}.plan] \
  --cache_path  [dla_int8_{policy}_s{seed}.cache] \
  --precision int8 \
  --use_dla 0
```

#### 2-C) Hybrid: GPU FP16 엔진(cut 이후 + head)

```bash
python3 build_engine_cli.py \
  --onnx_path [gpu_part.onnx] \
  --engine_path [gpu_part_fp16.plan] \
  --precision fp16
```

> Tip: 빌드 직후 binding name 확인

```bash
trtexec --loadEngine=[dla_int8_{policy}_s{seed}.plan] --dumpBindings
trtexec --loadEngine=[gpu_part_fp16.plan] --dumpBindings
```

### Step 3) Accuracy 평가(Track A) — COCO val2017 mAP 생성

#### 3-A) GPU-only FP16 (baseline)

```bash
./bench --mode=gpu [YOLOv11m_fp16.plan] [your/valid/set/dir] \
  --sched=throughput \
  --conf=0.001 --iou=0.45 --max_det=1000 \
  --pred_json pred_gpu_fp16.json
```

#### 3-B) Hybrid (DLA INT8 + GPU FP16)

```bash
./bench --mode=hybrid [dla.plan] [gpu.plan] [your/valid/set/dir] \
  --sched=throughput \
  --conf=0.001 --iou=0.45 --max_det=1000 \
  --connect "<DLA_OUT_P3>:<GPU_IN_P3>,<DLA_OUT_P4>:<GPU_IN_P4>,<DLA_OUT_P5>:<GPU_IN_P5>" \
  --pred_json pred_hybrid_int8fp16_s{seed}.json
```

> NOTE: `<DLA_OUT_*>`와 `<GPU_IN_*>` 텐서명은 엔진 빌드 시점의 binding name에 종속됩니다
> `trtexec --loadEngine=<plan> --dumpBindings` 결과로 이름을 확정한 뒤 `--connect`에 입력합니다.

COCO eval(예시):

```bash
python3 validation.py --gt gt.json --pred pred_hybrid_int8fp16_s{seed}.json --exclude_file exclude_ids.txt
```

### Step 4) PTQ 내부 산출물 분석 — cache(step) 비교 + 원인(히스토그램) 재현

#### 4-A) cache(step) diff (const vs mean/random)

```bash
python3 cache_diff_both_csv.py \
  --cache_a [a.cache] \
  --cache_b [b.cache] \
  --endian big --assume step \
  --keep [bits_neg/bits_pos] \
  --min_ratio [1.x] \
  --sort abs_bits --topk 10 \
  --out_csv [your/results/path.csv]
```

#### 4-B) 히스토그램 누적 + KL-sweep 근사 재현(선택)

* 동일 이미지셋에서 텐서(예: `/model.2/m.0/cv1/conv/Conv_output_0`)를 탭하여 abs hist를 누적
* KL(T) 곡선/`best_Thr` 및 `TV_norm/HFE_norm`를 산출해 const vs mean/random 비교

> NOTE: 구현 상세는 본문 4 및 Appendix D에 정리했습니다.

### Step 5) 성능/전력/에너지 벤치(Track B) — 동일 프로토콜로 비교

#### 5-A) GPU-only (throughput / latency)

```bash
./bench --mode=gpu [YOLOv11m_fp16.plan] [your/valid/set/dir] \
  --sched=throughput \
  --conf=0.25 --iou=0.45 --max_det=300 \
  --fps_cap=30 --fps_cap_mode=input \
  --power=[power_gpu.csv] --power_frames=[power_frames_gpu.csv] --power_key=VDD_IN --power_ms=n \
  --util=[util_gpu.csv] --util_ms=100 --util_py=jtop_export_v2.py --timing=[timing_gpu.csv]
```

#### 5-B) Hybrid (throughput / latency)

```bash
./bench --mode=hybrid [dla_int8.plan] [gpu_fp16.plan] [your/valid/set/dir] \
  --connect "<DLA_OUT_P3>:<GPU_IN_P3>,<DLA_OUT_P4>:<GPU_IN_P4>,<DLA_OUT_P5>:<GPU_IN_P5>" \
  --sched=throughput \
  --conf=0.25 --iou=0.45 --max_det=300 \
  --fps_cap=30 --fps_cap_mode=input/done \
  --power=[power_hybrid.csv] --power_frames=[power_frames_hybrid.csv] --power_key=VDD_IN --power_ms=n \
  --util=[util_hybrid.csv] --util_ms=100 --util_py=jtop_export_v2.py --timing=[timing_hybrid.csv]
```

#### 5-C) Idle baseline 측정 + Idle-subtracted 지표 산출

동일 환경에서 idle 측정 로그를 함께 수집한 뒤,

* `P_excess = P_avg - P_idle`
* `E_frame = P_excess / FPS`
  를 계산한다.

(예시)

```bash
python3 stats.py \
  --gpu_idle_power [power_gpu_idle.csv] \
  --gpu_load_power [power_gpu.csv] \
  --gpu_load_pframes [power_frames_gpu.csv] \
  --gpu_load_util [util_gpu.csv] \
  --gpu_load_timing [timing_gpu.csv] \
  --hybrid_idle_power [power_hybrid_idle.csv] \
  --hybrid_load_power [power_hybrid.csv] \
  --hybrid_load_pframes [power_frames_hybrid.csv] \
  --hybrid_load_util [util_hybrid.csv] \
  --hybrid_load_timing [timing_hybrid.csv] \
  --power_key p_w \
  --skip_frames 10 \
  --idle_skip_seconds n1 \
  --load_skip_seconds n2 \
  --util_join mean
```

* **재현 체크리스트(최소)**

* [ ] 운영 전처리(const114)는 유지, mean/random은 calibration 입력 생성 단계에만 적용되었는가
* [ ] seed sweep(42/84/126)에서 mean/random이 const 대비 mAP 회복을 보이는가
* [ ] cache ratio top 텐서(Top-3)가 재현되는가(허용 오차 범위 내)
* [ ] throughput/energy 지표가 동일 프로토콜로 산출되었는가(Idle-subtracted)

<br/>

## Appendix D. KL 최적점 근사 재현: Histogram → KL-sweep → Zigzag 지표

> 목적: 본문 4에서 주장한 “bin-mass spike/comb 구조가 KL 목적함수의 saw-tooth 성분과 best_Thr 선택이 padding 변화에 따라 달라질 수 있음"을 근사 재현하는 것입니다.
> 본 부록은 “증명”이 아니라, 동일 입력 셋에서 padding 정책(const vs mean/random)만 바꿨을 때
> (i) 히스토그램 bin-mass 배치와 (ii) KL(T) 곡선의 비매끄러움이 함께 바뀌며 (iii) best_Thr가 재현성 있게 이동하는지를 **관측 가능한 형태로 정리**하는 것이 목적입니다.

### D.0 핵심 설정(재현 전제)

* 동일 이미지셋: **1K calibration 이미지(동일 seed / 동일 파일 리스트)**
* 변경 변수: calibration 입력 생성 단계의 padding 정책만 변경
  * `const(114)` vs `mean` vs `random`
  
* 대상 텐서(대표 case study)
  * `/model.2/m.0/cv1/conv/Conv_output_0` (필요 시 다른 Top-3 텐서로 확장)
  
* 탭/추출 방식
  * ONNXRuntime(FP32) 또는 TensorRT debug/tap 등 **외부에서 텐서 값을 추출**해 offline histogram 누적

* 히스토그램 정의
  * `abs histogram`, `range=[0, 32]`, `bins=2048`

* KL-sweep 정의
  * threshold 후보 `T`(또는 `k`)를 스윕하며 원본 분포 `p`와 양자화 근사 분포 `q_T` 사이의 `KL(p||q_T)` 계산
  * `best_Thr = argmin_T KL(p||q_T)`

> NOTE: TensorRT calibrator의 exact 구현(클리핑/re-bin/스무딩 등)은 공개되지 않거나 버전에 따라 달라질 수 있으므로, 본 절은 “내부와 1:1 동일”을 목표로 하지 않습니다.
> 대신, bin-mass spike가 목적함수의 비연속성(톱니 성분)과 best_Thr 이동에 미치는 영향이 padding 변화만으로 재현되는지에 초점을 둡니다.

### D.1 데이터 수집: 텐서 값 추출(정책별)

정책별로 동일한 1K 입력을 통과시켜, 대상 텐서 출력 값을 파일로 수집합니다.

* 입력: `calib_1k_const.npy`, `calib_1k_mean.npy`, `calib_1k_random.npy`
* 출력(예시): `tap_const.memmap`, `tap_mean.memmap`, `tap_random.memmap`
* 저장 형태

  * 메모리 절약을 위해 `float16` 또는 `float32` memmap 권장
  * shape은 `(N, C, H, W)` 또는 flatten하여 `(N, D)` 가능

> NOTE: 결과 파일이 있다는 전제에서 분석/계산 절차를 정의합니다.

### D.2 히스토그램 누적(Abs hist) 및 기본 통계

정책별로 다음을 산출합니다.

1. 요약 통계(참고용)

* `q99`, `q99.9`,`q99.99` `amax`, `mean(abs)`,`std`, `clip_rate@T`(선택)

2. abs histogram

* `h[x]` = bin count (또는 density)
* `p = h / sum(h)` (정규화)

3. bin-mass spike/comb 징후(정책 간 비교)

* 상위 `Top-K bins by mass` (K=64 등) 비교
* `mass`가 집중된 bin 구간을 윈도우로 묶어 `mass_window` 비교(“merged windows” 방식)

> NOTE: 이 단계의 핵심은 range 변화가 아니라 **bin-mass 배치(이산화 결과)가 정책에 따라 달라지는지**를 기록하는 것입니다.

### D.3 KL-sweep(근사): q_T 구성 및 KL 계산

#### D.3.1 스윕 축 정의

* `T`는 abs domain에서의 클리핑 임계값(상한)으로 정의합니다.
* 히스토그램이 `[0, 32]` 범위에 있으므로, `T ∈ (0, 32]`를 적절한 간격으로 스윕합니다.

  * 예: `T = t_idx / bins * 32` 또는 `T`를 bin index(k)로 스윕

#### D.3.2 q_T(양자화 근사 분포) 생성(단순 근사)

정확한 TRT 구현 대신, 다음과 같은 표준적인 histogram-based PTQ 근사를 사용했습니다.

* `T` 이상 bin은 모두 `T` bin(또는 마지막 bin)으로 클램프
* `[0, T]` 구간을 `n_levels`(예: 128 또는 255 등)로 균등 양자화했다고 가정하고,

  * 원본 bin을 가장 가까운 quant bin에 매핑하여 재분배(또는 bucket 합산)
* 결과 분포를 다시 histogram bin space로 복원/업샘플링하여 `q_T` 구성
* `q_T`는 `eps`로 바닥값을 둔 후 정규화(0 나눗셈 방지)

> 본 근사 실험에서는 abs histogram(range=[0,32], bins=2048)를 고정한 상태에서 threshold T를 bin-grid 단위로 스윕합니다. 이때 const 조건에서는 특정 bin에 질량이 과도하게 집중(spike/comb)되어, T를 한 스텝 이동할 때 재분배되는 질량이 급격히 변하고 q_T가 계단형으로 크게 변하는 현상이 관측됩니다. 그 결과 KL(T) 곡선은 saw-tooth 형태로 요동하며 국소 저점이 증가합니다.

> 반면 mean/random에서는 bin-mass가 분산되어 동일한 스윕 조건에서도 q_T 변화가 완만해지고 KL(T)가 상대적으로 매끈해지는 경향이 재현됩니다.

> 동일한 스윕 조건에서 const에서만 계단형 변동이 크게 나타났으며, mean/random에서는 완화되는 경향이 재현되었습니다(동일 range/bins 고정).

#### D.3.3 KL 계산

* $KL(T) = Σ_i p_i * log(p_i / q_{T,i})$
* $best_{Thr} = argmin KL(T)$를 기록합니다.

### D.4 KL(T) 톱니 성분(zigzag) 정량화 지표(본문과 동일한 지표)

#### D.4.1 TV_norm (Total Variation, 정규화)

* $TV = Σ_t |KL(t+1) - KL(t)|$
* $TV_norm = TV / (max(KL) - min(KL) + ε)$

해석: `TV_norm`이 클수록 인접 T 이동에 따른 KL 변화가 거칠고, 톱니 성분이 크다.

#### D.4.2 HFE_norm (High-Frequency Energy, 정규화)

* 2차 차분: $d2(t) = KL(t+1) - 2*KL(t) + KL(t-1)$
* $HFE = Mean(|d2(t)|)$
* $HFE_{norm} = HFE/(q95(KL)-q5(KL))$

해석: `HFE_norm`이 클수록 고주파(톱니) 성분이 크다.

### D.5 그림: KL(T) overlay (const vs mean vs random)

**Figure D-1. KL(T) curve overlay with best_Thr markers (abs hist range=[0,32], bins=2048)**
![KL-curve](./assets/figures/kl-curve.png)

* 동일한 1K calibration 이미지 셋에서 padding 정책(const/mean/random)만 변경한 뒤,
  대상 텐서(예: `/model.2/m.0/cv1/conv/Conv_output_0`) 출력으로부터 abs histogram을 누적하고 KL-sweep을 수행하여 `KL(T)` 곡선을 계산합니다.
* 그림에는 세 정책의 `KL(T)`를 동일 축에 overlay하고, 각 정책에서 선택된 `best_Thr = argmin_T KL(T)`를 마커로 표시합니다.
* 관측 포인트

  * const는 `KL(T)` 곡선이 상대적으로 톱니(saw-tooth) 형태를 보이며 국소 저점이 다수 나타나는 경향
  * mean/random은 곡선이 상대적으로 매끈해지고, `best_Thr` 위치가 const 대비 이동하는 패턴 재현

> NOTE: 본 부록은 TRT calibrator 구현을 1:1로 “증명”하기 위한 것이 아니라,
> padding 정책 변화만으로도 KL 목적함수 형태 및 global optimum(`best_Thr`) 위치가 재현성 있게 달라질 수 있음을 시각적으로 기록하기 위한 근사 재현입니다.

### D.6 해석 가이드

* 본 절은 TRT calibrator의 상세 구현을 역공학적으로 “증명”하기 위한 것이 아니라,
  padding 정책 변화만으로도
  (i) bin-mass spike/comb 구조와 (ii) KL(T) 곡선의 톱니 성분이 동반 변화하며
  (iii) best_Thr(global optimum)가 재현성 있게 이동할 수 있음을 근사 실험으로 관측하는 데 목적을 둡니다.
* 따라서 결론은 “TRT 내부가 정확히 이렇다”가 아니라,
  **binning/이산화에 민감한 입력 조건(단색 반복 padding)이 KL 기반 threshold 선택을 불안정하게 만들 수 있다**는 점을 실험적으로 보이는 것입니다.
