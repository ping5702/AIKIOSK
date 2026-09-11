FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/myshell-ai/MeloTTS.git /app/melotts
WORKDIR /app/melotts

# CPU 전용 torch를 먼저 설치해서, melotts 설치 시 불필요한 GPU(CUDA) 의존성
# (nvidia-cublas-cu12 등 수백MB~수GB)가 딸려오는 것을 방지한다.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# melotts의 requirements.txt를 그대로 pip이 풀게 하면 cached_path -> boto3/
# google-cloud-storage 조합에서 수십 분씩 backtracking에 빠진다(둘 다 실제로는 안 씀,
# gradio 웹UI도 우리는 안 씀 - 우리 자체 Flask 서버를 씀). melotts 코드는 --no-deps로
# 설치하고, 한국어 합성에 실제로 필요한 패키지만 직접 골라 설치한다.
RUN pip install --no-cache-dir --no-deps -e .
RUN pip install --no-cache-dir --no-deps cached_path
RUN pip install --no-cache-dir \
    txtsplit \
    "transformers==4.27.4" \
    "num2words==0.5.12" \
    "jamo==0.4.1" \
    "g2pkk>=0.1.1" \
    "librosa==0.9.1" \
    "pydub==0.25.1" \
    "unidecode==1.3.7" \
    "anyascii==0.3.2" \
    "inflect==7.0.0" \
    "eng_to_ipa==0.0.2" \
    "unidic_lite==1.0.8" \
    "unidic==1.1.0" \
    "mecab-python3==1.0.9" \
    "fugashi==1.3.0" \
    "pykakasi==2.2.1" \
    "g2p_en==2.1.0" \
    "gruut[de,es,fr]==2.2.3" \
    "pypinyin==0.50.0" \
    "cn2an==0.5.22" \
    "jieba==0.42.1" \
    "langid==1.1.6" \
    tqdm \
    "tensorboard==2.16.2" \
    "loguru==0.7.2" \
    rich filelock

# cached_path의 schemes/__init__.py가 안 쓰는 GCS/S3 클라이언트도 무조건 import하므로
# (실제로는 안 쓰지만) import 성공을 위해 설치는 해야 한다. 이걸 다른 패키지들과
# 한꺼번에 풀면 backtracking에 빠지니, 이렇게 완전히 독립된 명령으로 설치한다.
RUN pip install --no-cache-dir "boto3==1.34.162" "google-cloud-storage==2.18.2"

RUN python -m unidic download
# 한국어만 미리 받아 이미지 레이어에 캐싱 (전체 언어 다운로드하는 init_downloads.py 대신).
# TTS(...) 생성자만 호출하면 메인 합성 모델만 받아지고, 실제 합성(tts_to_file) 때
# 지연 로딩되는 보조 BERT 모델(kykim/bert-kor-base)은 안 받아진다 - 반드시 실제로
# 한 번 합성까지 해봐야 그 모델과 관련 pip 의존성까지 전부 이미지에 캐싱된다.
RUN python -c "\
from melo.api import TTS; \
model = TTS(language='KR', device='cpu'); \
speaker_id = model.hps.data.spk2id['KR']; \
model.tts_to_file('안녕하세요.', speaker_id, '/tmp/warmup.wav', speed=1.0)"

# 모델은 위 단계에서 이미 캐시됐다. 이 시점부터 HF_HUB_OFFLINE=1을 켜야, 컨테이너가
# 실행될 때마다 server.py의 TTS(...) 생성자가 매번 HuggingFace Hub에 최신 버전인지
# 확인하러 나가서 느려지거나(오프라인 환경에서는 멈추는) 문제를 막을 수 있다.
ENV HF_HUB_OFFLINE=1

RUN pip install --no-cache-dir flask

COPY tts_server/server.py /app/melotts/server.py

EXPOSE 8890

CMD ["python", "server.py"]
