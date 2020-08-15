# Semi-SDP
Semi-supervised parser for semantic dependency parsing.

This repo contains the code used for the semi-supervised semantic dependency parser in Jia et al. (2020), [Semi-Supervised Semantic Dependency Parsing Using CRF Autoencoders](https://www.aclweb.org/anthology/2020.acl-main.607.pdf). 
Part of the codebase is extended from [Parser-v3](https://github.com/tdozat/Parser-v3).

## Requirement
```
python3
tensorflow-gpu=1.13.1
```
## How to use
### Training
Our semi-sdp parser can be trained by simply running
```
python3 main.py train UnlabelGraphParserNetwork  --force --config_file config/myconfig.cfg 
```
`config/myconfig.cfg` can be underspecified, with all default parameters being copied over. 
