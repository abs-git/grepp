## Description


### 각 단계별 문제점과 해결 방법

### 데이터 분할
제공 받은 데이터 셋은 train set이며 apple(127), cherry(127), tomato(50)개 입니다. <br>
학습을 위해 각 클래스별로 validation set(10), test set(10) 을 분할하여 평가에 사용하였습니다. <br>
test set은 onnx 변환 후에 적용됩니다. <br>

| Class    | Total Images | Number of Train | Number of Validation | Number of Test |
|----------|--------------|-----------------|----------------------|----------------|
| `apple`  | 127          | 107             | 10                   | 10             |
| `cherry` | 127          | 107             | 10                   | 10             |
| `tomato` | 50           | 30              | 10                   | 10             |


#### 학습 파이프라인
학습 파이프라인은 multi-gpu 기반의 **분산 학습**이 가능하도록 설계하였습니다.
0번 gpu (rank0) 에서 프로세스가 작업할 때, validation set에 대한 모델의 성능 평가와 grad-cam을 실행하도록 하였습니다.
`train/utils/engine.py`


#### 하이퍼파라미터 할당
하이퍼파라미터는 기본적으로 config 폴더 내에 yaml 형식으로 관리하였습니다. `config/base.yaml`
이를 기반으로 가변적인 파라미터로는 learning rate, optimizer, scheduler, batch size 를 선정하였고,
이들의 조합으로 **반자동화**를 하고자 하였습니다. `train/trainer.py`
하이퍼파라미터 조합에 따라 새로운 학습으로 구분하여 결과를 달리 저장하였습니다. `outputs/`


#### 모델 설계
기존의 모델은 3개의 convolution layer로 구성되어 5개의 클래스를 분류하는 모델입니다.
이를 기반으로 3개의 클래스를 분류하기 위해 기존 모델을 base model로 입력받는 End2End 모델을 구현하였습니다.
End2End 모델은 활성 함수와 head 부분을 추가하였습니다. `model/model.py`


#### 모델 평가시 지표, 방법
- Loss 함수
    - loss 함수는 multi class classifition을 위해 CrossEntropyLoss, FocalLoss를 구현해 적용하였습니다. `train/utils/loss.py`
    - 각 loss 값에 대한 가중치를 부여하기 위해 focal loss에 대한 ratio 적용하였습니다.

- Metric 적용
    - 또한, 정밀한 학습 결과를 확인하기 위해 모델의 예측 값과 실제 값으로 FP, FN, TP, TN를 계산하는 함수를 추가하였으며, <br> Precision, recall, f1-score, miss rate, accuracy를 그래프 형태로 저장하여 **정량적인** 모델의 성능을 평가하였습니다. `train/utils/metric.py`

- GradCam 생성
    - 3개의 층으로 구성된 convolution layer를 hook으로 등록하여 일정 주기마다 현재 모델 상태에 대한 feature map을 추출하여 grad-cam을 생성하였습니다. `train/utils/grad.py`
    - grad-cam을 기반으로 모델이 올바른 feature에 대해 최적화가 이루어지는지를 고려해 **정성적인** 모델의 성능의 성능을 평가하였습니다.


### 양자화

- ONNX 변환
모델 가속화를 위해 End2End 모델과 학습된 weights 파일을 기반으로 onnx 변환을 하였습니다.
우선, 원본 모델을 fp32에 대한 onnx 모델 변환 후 int8 모델로 변환하여 검증하였습니다. `deploy/deploy-onnx.py`
모델의 성능은 test set을 기반으로 정확도(accuracy)를 측정하였습니다. `deploy/test-onnx.py`


### Insight

- batch size
    - batch size 4, 8, 16 을 부여했을 때 batch size의 크기가 클수록 학습에 유리합니다. (output_0 vs output_4 vs output_8)

- optimizer
    - optimizer는 adam가 sgd보다 최적화에 유리합니다. (output_8 vs output_10)

- scheduler
    - scheduler는 steplr이 exponentiallr 보다 빠른 수렴이 가능합니다. (output_8 vs output_9)

- learning rate
    - learning rate는 0.0001 보다 작은 값에서 학습이 이루어지는 것을 확인했습니다.
