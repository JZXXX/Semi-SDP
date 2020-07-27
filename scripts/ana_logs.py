#coding=utf-8
import os
import re


def printsortedDict(adict, obj, reversed=True):

    sortedDict = sorted(adict.items(), key=lambda x:x[1], reverse=reversed)
    for (key, value) in sortedDict:
        obj.write(key + '\t' + 'result=' + str(value) +'\n')
    


def ana_logs(filepath, outputfile, prefix):
    #filepath: log fold e.g. './log-829'; outputfile: result file name e.g.: 'output.txt'; prefix: e.g. 'DM19-semi'
    records = {}
    for root, _, files in os.walk(filepath):
        for filename in files:
            if filename.startswith(prefix):
                print('filename', filename)
                with open(root + '/' + filename) as f:
                    lines = f.readlines()
                    line = lines[-3].strip()
                    print('line',line)
                    if line.startswith('Current best UF1'):
                        print('line', line)
                        line = line.split(':')
                        print('split line', line)
                        result = float(line[1])
                        # tmp.set_result(result)
                        records[filename] = result
                        print('result')
                    else:
                        raise ValueError
        # L.sort(key=lambda log_rec:log_rec.result, reverse=True)
        output_filepath = filepath + '/' + outputfile
        f = open(output_filepath, 'w')
        printsortedDict(records, f, True)
    # for i in L:
    #     print(i)
    #     print(i, file=f)
    
    # print('best result: ', L[0])

if __name__== '__main__':
    import sys
    # ana_logs(filepath='./log-829', outputfile='semi5.5_blurdm.txt', prefix='DM19-semi')
    ana_logs(sys.argv[1], sys.argv[2], sys.argv[3])