from functools import reduce
import os
import pdb
def gen_command(template_path,times):
    base_str = 'CUDA_VISIBLE_DEVICES=3 nohup python3 main.py train UnlabelGraphParserNetwork  --force --config_file'
    write_file = template_path.rstrip('.cfg')+('.command.txt')
    with open(write_file,'w') as fw:
        for i in range(times):
            config = template_path.rstrip('.cfg')+'-'+str(i)+('.cfg')
            log_file = 'log'+config.lstrip('config').rstrip('.cfg')+'.log'
            fw.write(base_str+' '+config+' '+'>'+log_file+'&'+'\n')
def aba_config(template_path):
    write_dir, file_name = os.path.split(template_path)
    type = ['basic','char','char+lemma']
    # write_files = []
    unlabel = False

    for i in type:
        # write_files.append(os.path.join(write_dir,file_name.split('-')[0]+'-'+'i'))
        write_file = os.path.join(write_dir,file_name.split('-')[0]+'-'+file_name.split('-')[1]+'-'+i)
    # for write_file in write_files:
        with open(write_file+'.cfg', 'w') as fw:
            with open(template_path) as fr:
                for line in fr:
                    if line.strip()=='[UnlabelGraphParserNetwork]':
                        unlabel = True
                    co = line.split('=')[0].strip()
                    if co == 'save_metadir':
                        fw.write('save_metadir = saves/' + file_name.split('-')[0] + '/'+file_name.split('-')[1] + '/'+i+'\n')
                    elif co == 'input_vocab_classes' and unlabel:
                        if i == 'basic' or 'char':
                            fw.write('input_vocab_classes = FormMultivocab:UPOSTokenVocab'+'\n')
                            unlabel = False
                        elif i == 'char+lemma':
                            fw.write('input_vocab_classes = FormMultivocab:UPOSTokenVocab:LemmaTokenVocab' + '\n')
                            unlabel = False
                    elif co == 'use_subtoken_vocab':
                        if 'char' in i:
                            fw.write('use_subtoken_vocab = True' + '\n')
                        else:
                            fw.write(line)
                    else:
                        fw.write(line)
    # gen_command(template_path,int(times))

if __name__ == '__main__':
    """"""

    import sys
    aba_config(sys.argv[1])