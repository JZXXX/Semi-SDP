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
python3 main.py train UnlabelGraphParserNetwork  --force --config_file $CONFIGFILE
```
By default, if the save directory already exists, you'll get a prompt warning you that the system will delete it if you continue and giving you one last chance to opt-out. If you are debugging or want to run the program in the background, add the `--force` flag. 

### Parsing
The trained model can be run by calling
```
python main.py --save_dir $SAVEDIR run --output_dir TestResult $DATAFILE
```
This will save parsed sentences in `DATAFILE` to the `TestResult/` directory--make sure no files in different directories have the same basename though, or one will get overwritten! The sentences in `DATAFILE` is in `CoNLL-U` format. `SAVEDIR` is the directory that your trained model saved.
