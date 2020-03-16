#FROM pytorch_video
FROM cu90-py35-trt5-torch41-acap-mkl
USER root
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/TensorRT-5.0.0.10/lib:/usr/local/cuda/lib64/:/usr/local/cuda/doc/man/man7/:/usr/local/lib/python3.5/dist-packages/torch/lib/
ENV PATH=$PATH:/usr/loca/cuda/bin/

###############################################
# 以上部分用户可定制，以下部分不可删除
###############################################
# 项目构建
WORKDIR /data
COPY . .
# 指定启动路径
#ENTRYPOINT cd test && python run.py
