FROM nvidia/cuda:11.0-base
CMD python3 web.py
RUN apt update && apt install -y python3 python3-pip wget git zstd curl && pip3 install torch
RUN wget -c https://mystic.the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
RUN git clone https://github.com/kingoflolz/mesh-transformer-jax.git
RUN pip3 install -r mesh-transformer-jax/requirements.txt
RUN pip3 install mesh-transformer-jax/ jax==0.2.12 jaxlib==0.1.68 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN tar -I zstd -xf step_383500_slim.tar.zstd &&\
  mkdir gpt-j-hf &&\
  curl https://raw.githubusercontent.com/kingoflolz/mesh-transformer-jax/master/configs/6B_roto_256.json > gpt-j-hf/config.json
COPY converttotorch.py ./
RUN python3 converttotorch.py
RUN pip install cherrypy
COPY test.py ./
COPY web.py ./
COPY model.py ./


