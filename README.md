## AllenNLP

```text
https://guide.allennlp.org/

https://allenai.org/allennlp
https://github.com/allenai/allennlp
https://github.com/MaksymDel/allennlp-light
https://github.com/flairNLP/flair
https://allenai.org/papers?tag=AllenNLP


```


```text
cd /home/honeytian/

git clone https://github.com/qgyd2021/AllenNLP.git

docker build -t allennlp:v20260715_1829 .

docker stop allennlp_7865 && docker rm allennlp_7865

docker run -d \
--name allennlp_7865 \
--network host \
--restart always \
--gpus all \
-e server_port=7865 \
allennlp:v20260715_1829

docker run -itd \
--name allennlp_7865 \
--network host \
--restart always \
--gpus all \
-e server_port=7865 \
allennlp:v20260715_1829 /bin/bash


http://192.168.34.115:7865/

```
