def gen_command(read_file):
    write_file = read_file.split('/')[-1].strip('new_configs.txt')+'new_commands.txt'
    base_str = 'CUDA_VISIBLE_DEVICES=3 nohup python3 main.py train UnlabelGraphParserNetwork  --force --config_file'
    with open(write_file,'w') as fw:
        with open(read_file) as fr:
            for line in fr:
                line = line.strip()
                log_file = line.split('/')[1].strip('.cfg')
                type = line.split('/')[0].split('-')[1]
                fw.write(base_str+' '+line+' '+'>'+type+'-'+log_file+'&'+'\n')
if __name__ == '__main__':
    """"""

    import sys
    gen_command(sys.argv[1])
