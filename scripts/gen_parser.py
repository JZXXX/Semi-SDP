import os
def gen_parser(dir):
    if 'DM' in dir:
        write_file = 'parse_dm.sh'
        data = 'data/AllData/DM/test.en.id.dm.conllu data/AllData/DM/test.en.ood.dm.conllu'
    elif 'PAS' in dir:
        write_file = 'parse_pas.sh'
        data = 'data/AllData/PAS/test.en.id.pas.conllu data/AllData/PAS/test.en.ood.pas.conllu'
    elif 'PSD' in dir:
        write_file = 'parse_psd.sh'
        data = 'data/AllData/PSD/test.en.id.psd.conllu data/AllData/PSD/test.en.ood.psd.conllu'
    with open(write_file,'w') as fw:
        fw.write('echo ' + dir + '\n')
        for name in os.listdir(dir):
            fw.write('echo '+name + '..................................'+'\n')
            fw.write('sh parse.sh ' + dir+'/'+name + ' TestResult ' + data + '\n')

if __name__ == '__main__':
    """"""

    import sys
    gen_parser(sys.argv[1])