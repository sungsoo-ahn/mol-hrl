FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update
RUN apt-get install -y git

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit

RUN pip install guacamol
RUN pip install git+https://github.com/bp-kelley/descriptastorus
RUN pip install chemprop
RUN conda install -c conda-forge scikit-learn==0.21.3

RUN pip install pytorch-lightning
  
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
RUN pip install torch-geometric

ENV NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ=="
