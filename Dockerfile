FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update
RUN apt-get install -y git

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit

RUN pip install pytorch-lightning
  
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu111.html
RUN pip install torch-geometric

ENV NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ=="
