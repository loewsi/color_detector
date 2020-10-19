FROM duckietown/dt-duckiebot-interface:daffy-arm32v7

WORKDIR /color_detector

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV N_SPLITS=5

COPY color_detector.py .

CMD python3 ./color_detector.py
