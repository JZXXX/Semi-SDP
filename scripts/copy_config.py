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
def copy_config(template_path,times):
    for i in range(int(times)):
        write_file = template_path.rstrip('.cfg')+'-'+str(i)
        with open(write_file+'.cfg', 'w') as fw:
            with open(template_path) as fr:
                for line in fr:
                    co = line.split('=')[0].strip()
                    if co == 'save_metadir':
                        fw.write(line.strip()+'-'+str(i)+'\n')
                    else:
                        fw.write(line)
    gen_command(template_path,int(times))

if __name__ == '__main__':
    """"""

    import sys
    copy_config(sys.argv[1],sys.argv[2])
    # import os
    # for filename in os.listdir('config/ablation/semi'):
    #     if 'semi' in filename:
    #         copy_config('config/ablation/semi/'+filename, 3)