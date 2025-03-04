# Dockerfile for photorealistic visual SLAM test video generation

# ベースイメージ: Python 3.9 (slim-buster: 軽量版)
FROM python:3.9-slim-buster

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール (FFmpeg, libgl1)
# libgl1: OpenGL関連 (Pillowが画像処理で必要とする場合がある)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx

RUN apt-get install -y libgl1-mesa-dev

# 必要なPythonライブラリをインストール
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update -y
RUN apt-get install -y git
RUN git clone https://github.com/mmatl/pyglet.git 
RUN cd pyglet && pip install .


RUN pip install numpy Pillow pyquaternion imageio
RUN pip install trimesh pyrender glfw PyOpenGL

# スクリプトをコピー
# COPY vslam_test.py .

# テクスチャファイルをコピーする場所を作成 (オプション)
# テクスチャファイルを使用する場合は、このディレクトリにコピーする
#RUN mkdir textures

# 最後に不要なファイルを削除してイメージサイズを削減
RUN rm -rf /var/lib/apt/lists/*

# 実行コマンド (コンテナ起動時に実行される)
# CMD ["python", "vslam_test.py"]

CMD ["/bin/bash"]