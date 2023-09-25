# stylegan3-edit-api

A simple API for editing images using StyleGAN3 in specific directions.

### Setup

insatll requirements

```
pip install -r requirements.txt

wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip ninja-linux.zip -d /usr/local/bin/
update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

and download pretrained models

```
gdown 12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-
mkdir pretrained_models
mv restyle_pSp_ffhq.pt pretrained_models
```

### Start server

```
python main.py
```
