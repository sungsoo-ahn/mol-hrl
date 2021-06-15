FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit
RUN conda install -y -c conda-forge scikit-learn 

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-geometric

RUN pip install joblib>=0.12.5
RUN pip install numpy>=1.15.2
RUN pip install scipy>=1.1.0
RUN pip install tqdm>=4.26.0
RUN pip install h5py==2.10.0
RUN pip install flake8>=3.5.0
RUN pip install mypy>=0.630
RUN pip install pytest>=3.8.2
RUN pip install guacamol --no-dependencies

RUN pip install git+https://github.com/bp-kelley/descriptastorus
RUN pip install chemprop
RUN conda install -c conda-forge scikit-learn==0.21.3