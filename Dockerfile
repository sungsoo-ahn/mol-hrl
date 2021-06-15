FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit

RUN pip install guacamol
RUN pip install git+https://github.com/bp-kelley/descriptastorus
RUN pip install chemprop
RUN conda install -c conda-forge scikit-learn==0.21.3