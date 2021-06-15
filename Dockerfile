FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update
RUN apt-get install git

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit

RUN pip install guacamol
RUN pip install git+https://github.com/bp-kelley/descriptastorus
RUN pip install chemprop
RUN conda install -c conda-forge scikit-learn==0.21.3

RUN export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ=="
