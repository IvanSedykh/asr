# ASR project barebones

## TL;DR
Final scores are:

### test-other
```
WER (ARGMAX): 31.91%
WER (BS): 31.62%
WER (BS + LM): 24.69%
```

### test-clean
```
WER (ARGMAX): 13.20%
WER (BS): 12.99%
WER (BS + LM): 9.39%
```

Note, that we even didn't use train-other.

Check WANDB report [here](https://wandb.ai/idsedykh/asr_project/reports/ASR-Conformer---Vmlldzo1NzM2NDMw?accessToken=8x10ypm1cv3gvk1a404po2265kpmty1r403fc2kideuwq5uyznv2m0r61kkl3xho).


## Installation guide

1. install conda, create new env
2. install torch
   ``` shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
3. install other stuff

   ```shell
   pip install -r ./requirements.txt
   ```
4. pray and hope



## Reproduction guide


run `python train.py -c hw_asr/configs/conformer_24.json`

To compute metrics:

Download the checkpoint, config and lm from [here](https://disk.yandex.ru/d/ZqIdzXFTsDvtEw).  
Or from gdisk.

`gdown --fuzzy https://drive.google.com/file/d/1Dn2ed35w9HbU3ouREGi-sE1RacwB182i/view?usp=sharing`
`unzip default_test_model.zip`

test-other:
```bash
python test.py \
    --batch-size 32 \
    --jobs 8 \
    -c default_test_model/config-other.json \
```


test-clean:
```bash
python test.py \
    --batch-size 32 \
    --jobs 8 \
    -c default_test_model/config-clean.json \
```

I have used a pretty powerful server, so it may fail in case of weaker machine.