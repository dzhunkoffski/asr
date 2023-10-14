# asr
dl-audio homework

## Install LMs

```bash
wget https://us.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gunzip -c 3-gram.pruned.1e-7.arpa.gz > hw_asr/text_encoder/language_models/3-gram.pruned.1e-7.arpa
wget https://us.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz
gunzip -c 3-gram.pruned.3e-7.arpa.gz > hw_asr/text_encoder/language_models/3-gram.pruned.3e-7.arpa
```
