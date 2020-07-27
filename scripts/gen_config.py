from functools import reduce
import os
import pdb
def gen_command(read_file,write_file):
    base_str = 'CUDA_VISIBLE_DEVICES=3 nohup python3 main.py train UnlabelGraphParserNetwork  --force --config_file'
    lines = []
    with open(write_file,'w') as fw:
        with open(read_file) as fr:
            for line in fr:
                line = line.strip()
                log_file = line.split('/')[-1].strip('.cfg')
                type = line.rstrip(line.split('/')[-1]).lstrip(line.split('/')[0])
                # pdb.set_trace()
                if not os.path.exists('log'+type):
                    os.makedirs('log'+type)
                lines.append(line)
                fw.write(base_str+' '+line+' '+'>'+'log'+type+log_file+'.log'+'&'+'\n')
            fw.write('\n')
            for line in lines:
                fw.write('CUDA_VISIBLE_DEVICES=3 sh parse.sh '+line.lstrip('config/').rstrip('.cfg')+' TestResult data/AllData/DM/test.en.id.dm.conllu data/DM-ood/test.en.dm.ood.half925.conllu' + '\n')



def gen_config(template_path, *parameter_str):
    # parameter_str: 'lr_0.1,0.01', 'bs_1000,800', 'us_0.1,0.2'....
    pa2con = {'lr': 'learning_rate',
              'bs':  'batch_size',
              'us':  'unsup_strength',
              'vc':  'max_embed_count',
              'drs': 'decoder_recur_size',
              'mhs':   'mlp_hidden_sizes'
              }

    write_dir = os.path.dirname(template_path)
    pre = ''
    if 'DM19-semi-' in os.path.basename(template_path):
        pre = 'DM19-semi-'
    elif 'PAS19-semi-' in os.path.basename(template_path):
        pre = 'PAS19-semi-'
    elif 'PSD19-semi-' in os.path.basename(template_path):
        pre = 'PSD19-semi-'
    pre = template_path.split('/')[-1].strip('.cfg')
    group = []
    paras = list(parameter_str)
    para_name = []
    for i in paras:
        pa, va = i.split('_')
        para_name.append(pa)
        va = list(va.split(','))
        para = []
        for n in va:
            para.append(pa+n)
        group.append(para)
    comb = lambda x : reduce(lambda x,y : [i+'_'+j for i in x for j in y], x)
    filename = comb(group)
    change_config = {}
    with open(pre+'new_configs.txt','w') as fwn:
        for fn in filename:
            fn_pre = pre+'-'+fn
            # pdb.set_trace()
            write_file = write_dir+'/'+fn_pre
            fwn.write(write_file+'.cfg'+'\n')
            fns = fn.split('_')
            pa_id = 0
            for i in fns:
                # pdb.set_trace()
                key = para_name[pa_id]
                value = i.strip(key)
                pa_id += 1
                change_config[pa2con[key]] = value
            with open(write_file+'.cfg', 'w') as fw:
                with open(template_path) as fr:
                    for line in fr:
                        co = line.split('=')[0].strip()
                        if co == 'save_metadir':
                            fw.write(line.split('=')[0]+'= saves/'+write_file.lstrip('config/')+'\n')
                        elif co in change_config:
                            fw.write(co+' = '+change_config[co]+'\n')
                        else:
                            fw.write(line)
    gen_command(pre+'new_configs.txt', write_dir+'/'+pre+'-new_commands.txt')

if __name__ == '__main__':
    """"""

    import sys
    gen_config(sys.argv[1], *sys.argv[2:])