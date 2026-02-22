

# hybrid-inference-experiments
# 하이브리드 추론 실험 정리 (현재 단계)
> **Target HW (as of YYYY-MM-DD): Jetson Orin NX 16GB**
> 초기 계획은 Orin Nano였으나, **Orin Nano에는 DLA가 없어** 본 프로젝트의 핵심인 **DLA(INT8)+GPU(FP16) 하이브리드 분할 실행**을 위해 Orin NX로 전환했습니다.
> ## Docs
- [상세 보고서](./YOLOv11m_Hybrid_Inference_on_Jetson_Orin_NX.md)
> ⚠️ 본 문서는 모바일 환경에서의 하이브리드 추론 실험을 정리한 기록으로, Orin Nano 기반 실험 이전 단계의 관찰과 제약 사항을 정리한 문서입니다.

---

# 1. 프로젝트 진행 배경

최근 모바일 기기에는 CPU, GPU, NPU 등 여러 가속기가 탑재되어 있으며, 이를 활용한 추론 실행이 가능해졌습니다. 본 프로젝트는 특정 모바일 기기에서 추론 실험을 진행하는 과정에서 시작되었습니다.

초기에는 개별 가속기의 동작 여부와 추론 실행 가능 여부를 확인하는 데 초점을 두었습니다. 그러나 실험을 진행하면서 가속기 선택과 실행 방식이 전체 실행 구조에 직접적인 영향을 미친다는 점을 인지하게 되었습니다.

<br/><br/>

# 2. 사전 실험 및 문제 인식

S21+ 및 S25 Ultra 환경에서 추론 실험을 진행한 결과, use_gpu=true 인자를 준 실행에서는 모델 그래프 전체가 GPU로 위임되어 실행되었으며 반복 실행 과정에서도 안정적인 추론이 가능했습니다.[A] 

반면 공식적으로 제공되는 `nnapi=true` 인자를 통해 NPU 사용을 시도한 경우에는, 모델 내부에 포함된 NPU 미지원 연산 또는 벤더 드라이버(예: eden-drv)의 모델 인식 제약으로 인해 대부분의 연산이 CPU로 fallback되었습니다. 또한 일반 사용자 권한에서 가속기 할당 방식을 명시적으로 지정할 수 있는 방법은 확인할 수 없었습니다.[B][C]

<br/><br/>

# 3. 문제 재정의

본 프로젝트의 목적은 NPU와 GPU를 동시에 사용하는 구조를 구현하는 데 그치지 않습니다. 특정 환경에서 하이브리드 추론을 시도하는 과정에서 발생하는 제약을 정리하고, 이러한 제약이 성능에 어떤 영향을 미치는지를 중심으로 살펴보는 데 초점을 둡니다.

즉 하이브리드 분할 추론이 이론적으로 가능한 구조인지 여부가 아니라, 실제 환경에서 실험 자체가 성립 가능한지를 먼저 확인하는 것을 본 단계의 문제로 정의합니다.

<br/><br/>

# 4. 설계 범위 및 향후 계획
하이브리드 추론 실험이 상용 안드로이드 기기에서 어려운 이유에 대한 분석을 하였습니다. 성능 향상 여부에 대한 판단은 가속기 지정과 실행 방식을 명확히 제어할 수 있는 Jetson Orin Nano 플랫폼을 활용하여 이후 진행할 예정입니다. 주제를 확장하고 실험 조건을 명확히 설정하여 하이브리드 추론 성능을 점검할 예정입니다.

<br/><br/>

# 5. 플랫폼 전환 및 향후 계획

하이브리드 추론 실험이 상용 안드로이드 기기에서 어려운 이유에 대한 분석을 하였습니다. 성능 향상 여부에 대한 판단은 가속기 지정과 실행 방식을 명확히 제어할 수 있는 Jetson Orin Nano 플랫폼을 활용하여 이후 진행할 예정입니다. 주제를 확장하고 실험 조건을 명확히 설정하여 하이브리드 추론 성능을 점검할 예정입니다.

<br/><br/>

## 부록 A. GPU 실행 로그 요약 (S21+)

- 실행 도구: TensorFlow Lite `benchmark_model`
- 모델: `yolo11n_float16.tflite`
- Delegate: `TfLiteGpuDelegateV2`
- Graph delegation: 전체 노드 GPU 위임 (546 / 546)
- CPU fallback: 발생하지 않음
- 평균 추론 시간: 약 **28.5 ms** (100 runs 기준)

<br/>

## 참고 로그 스니펫

```text
VERBOSE: Replacing 546 out of 546 node(s) with delegate (TfLiteGpuDelegateV2)
INFO: Explicitly applied GPU delegate, and the model graph will be completely executed by the delegate.
```

<br/><br/>

## 부록 B. NNAPI(`nnapi=true`) 인자를 사용하여 NPU 기반 추론을 시도한 결과에 대한 근거 1

- `nnapi=true` 옵션을 사용한 실행에서는, 사용 가능한 NNAPI 가속기 목록에 `nnapi-reference`(CPU 구현)만이 노출되고 하드웨어 가속기(NPU)에 대한 명시적 위임은 확인할 수 없었습니다.

- 실행 로그 상에서, 모델에 포함된 일부 연산이 NNAPI/NPU에서 지원되지 않거나, 벤더 드라이버(eden-drv)가 모델을 정상적으로 인식하지 못하는 경우가 관찰되었고, 그 결과 대부분의 연산이 CPU로 fallback되었습니다.

<br/>

## 부록 C. NNAPI(`nnapi=true`) 인자를 사용하여 NPU 기반 추론을 시도한 결과에 대한 근거 2
Android 플랫폼 및 벤더의 공식 문서를 확인하였습니다.

- Android 공식 NNAPI 문서에 따르면, "하드웨어 가속기 사용은 기기 제조사(OEM)가 제공하는 전문 공급업체(vendor) 드라이버에 의존하며, 해당 드라이버가 존재하지 않거나 노출되지 않은 경우 NNAPI 런타임은 CPU에서 요청을 실행한다" 라고 서술되어 있습니다.
  (Android Neural Networks API 문서 참고)

- 또한 NNAPI는 Android 15부터 deprecated(지원 중단 예정) 상태로 명시되어 있으며, 이는 앞으로도 하드웨어 가속기 제어 권한이 제공되지 않을 것임을 뜻합니다.
  (NNAPI Migration Guide 참고)

- TensorFlow Lite 공식 NNAPI delegate 문서에서는, NNAPI delegate 사용 시 지원되지 않는 연산이 CPU로 fallback될 수 있음을 설명하고 있으나, 이를 회피하기 위한 방법으로 연산 단위 또는 레이어 단위로 하드웨어 가속기를 사용자가 지정할 수 있는 방법은 제공하지 않습니다.

- Qualcomm QNN(AI Runtime) 공식 문서에서는, Snapdragon 플랫폼의 AI Engine 하드웨어에서 추론을 수행하기 위해서는 사전 학습된 모델을 QNN AI Runtime에서 사용 가능한 형식으로 변환(convert)해야 한다고 명시하고 있습니다.

위 문서들 내용을 종합할 때 일반 사용자 권한으로 애플리케이션이 Android 표준 API(NNAPI)를 통해 NPU를 직접 제어하거나, 가속기 할당 방식을 명시적으로 지정하는 공식적인 경로는 확인할 수 없었고, 앞으로의 지원 계획을 명확히 밝히고 있지 않습니다.

<br/>

### 참고 문서
- Android Neural Networks API (NNAPI)
  https://developer.android.com/ndk/guides/neuralnetworks?hl=ko

- NNAPI Migration Guide
  https://developer.android.com/ndk/guides/neuralnetworks/migration-guide?hl=ko

- TensorFlow Lite NNAPI Delegate
  https://www.tensorflow.org/lite/android/delegates/nnapi?hl=ko

- Qualcomm QNN AI Runtime Workflow
  https://docs.qualcomm.com/doc/80-62010-1KO/topic/qnn-workflow.html
