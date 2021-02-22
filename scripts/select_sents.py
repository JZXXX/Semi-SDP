def select_sents_length(path, upper_len):
    sent_len = 0
    sent_id = 0
    sent = []
    w_path = path.strip('conllu')+str(upper_len)+'.conllu'
    with open(w_path, 'w') as fw:
        with open(path, 'r') as fr:
            for line in fr:
                if line.strip():
                    sent_len += 1
                    sent.append(line)
                else:
                    # sentence has id #2000001
                    if sent_len-1 <= upper_len:
                        # fw.write('#' + str(sent_id) + '\n')
                        sent_id += 1
                        for i in sent:
                            fw.write(i)
                        fw.write('\n')
                    sent = []
                    sent_len = 0
    print(sent_id)

def select_sents_num(path, num):
    n = 0
    w_path = path.strip('conllu')+str(num)+'.conllu'
    with open(w_path, 'w') as fw:
        with open(path, 'r') as fr:
            for line in fr:
                if line.startswith('#'):
                    n += 1
                    if n > num:
                        break
                fw.write(line)

def concat(file1, file2):
    w_file = file1.strip('conllu')+file2.split('/')[-1]
    with open(w_file, 'w') as fw:
        with open(file1) as fr1:
            for line in fr1:
                fw.write(line)
        with open(file2) as fr2:
            for line in fr2:
                fw.write(line)


if __name__ == '__main__':
  
    # select_sents_length('data/DM/train.en.dm.conllu', 60)
 
 

