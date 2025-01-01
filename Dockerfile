FROM gcr.io/kaggle-gpu-images/python:latest

#言語と地域の設定
ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

#ライブラリのインストール
WORKDIR /kaggle
#各々のGPUに対応するpytorchをインストールhttps://pytorch.org/get-started/previous-versions/
RUN pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
ADD requirements.txt /kaggle/requirements.txt
RUN pip install -r requirements.txt

#jupyter notebookの起動
ADD run.sh /opt/run.sh
RUN chmod 700 /opt/run.sh
CMD /opt/run.sh
