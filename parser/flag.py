from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import os
import re
import codecs
import zipfile
import gzip
try:
  import lzma
except ImportError:
  try:
    from backports import lzma
  except ImportError:
    import warnings
    warnings.warn('Install backports.lzma for xz support')
from collections import Counter
import  tensorflow as tf
import numpy as np
#------------------------------------------------------
class Flag():


    # =============================================================
    def __init__(self, placeholder_shape=[None], config=None, isTrain=True, isDev=False):
        """"""

        self.placeholder = tf.placeholder(tf.int64, placeholder_shape, name=self.classname)

        self._config = config
        self.id2flag = {}
        self.index2flag = {}
        if isTrain:
            if self.fix:
                self.n_sent = self.set_flag_train_fixlabeldata()
            else:
                self.n_sent = self.set_flag_train()
                if self.portion > 1:
                    self.split()
        # else:
        #     self.n_sent = self.set_flag_dev()
        elif isDev:
            self.n_sent = self.set_flag_dev()
        else:
            self.n_sent = self.set_flag_test()
        return

    # =============================================================


    # =============================================================
    def set_placeholders(self, indices, feed_dict={}):
        """"""
        flag = self.indices2flag(indices)
        feed_dict[self.placeholder] = flag
        return feed_dict

    # =============================================================

    # =============================================================
    def indices2flag(self, indices):
        """"""

        return [self.index2flag[index] for index in indices]

    # =============================================================
    
    
    
    # =============================================================
    def set_flag_train_fixlabeldata(self):
        """"""
        num = 0

        if self.train_conllu_file[0].endswith('.zip'):
            open_func = zipfile.Zipfile
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.gz'):
            open_func = gzip.open
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.xz'):
            open_func = lzma.open
            kwargs = {'errors': 'ignore'}
        else:
            open_func = codecs.open
            kwargs = {'errors': 'ignore'}

        with open_func(self.train_conllu_file[0], 'rb') as f:
            reader = codecs.getreader('utf-8')(f, **kwargs)
            for line in reader:
                line = line.strip()
                if line and line.startswith('#'):
                    # if re.match('[0-9]+[-.][0-9]+', line):
                    if num < self.labeled_num:
                        self.id2flag[line] = 1
                        self.index2flag[num] = 1
                    else:
                        self.id2flag[line] = 0
                        self.index2flag[num] = 0
                    num += 1
        self.dump()
        return num
    # =============================================================
    
    

    # =============================================================
    def set_flag_train(self):
        """"""
        num = 0

        if self.train_conllu_file[0].endswith('.zip'):
            open_func = zipfile.Zipfile
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.gz'):
            open_func = gzip.open
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.xz'):
            open_func = lzma.open
            kwargs = {'errors': 'ignore'}
        else:
            open_func = codecs.open
            kwargs = {'errors': 'ignore'}

        with open_func(self.train_conllu_file[0], 'rb') as f:
            reader = codecs.getreader('utf-8')(f, **kwargs)
            for line in reader:
                line = line.strip()
                if line and line.startswith('#'):
                    # if re.match('[0-9]+[-.][0-9]+', line):
                    if num%self.portion == 0:
                        self.id2flag[line] = 1
                        self.index2flag[num] = 1
                    else:
                        self.id2flag[line] = 0
                        self.index2flag[num] = 0
                    num += 1
        self.dump()
        return num
    # =============================================================

    # =============================================================
    def set_flag_dev(self):
        """"""
        num = 0

        if self.dev_conllu_file[0].endswith('.zip'):
            open_func = zipfile.Zipfile
            kwargs = {}
        elif self.dev_conllu_file[0].endswith('.gz'):
            open_func = gzip.open
            kwargs = {}
        elif self.dev_conllu_file[0].endswith('.xz'):
            open_func = lzma.open
            kwargs = {'errors': 'ignore'}
        else:
            open_func = codecs.open
            kwargs = {'errors': 'ignore'}

        with open_func(self.dev_conllu_file[0], 'rb') as f:
            reader = codecs.getreader('utf-8')(f, **kwargs)
            for line in reader:
                line = line.strip()
                if line and line.startswith('#'):
                    # if re.match('[0-9]+[-.][0-9]+', line):
                    self.id2flag[line] = 0
                    self.index2flag[num] = 0
                    num += 1
        return num
    # =============================================================

    #------------------test---------------------------------------
    # =============================================================
    def set_flag_test(self):
        """"""
        num = 0

        if self.dev_conllu_file[0].endswith('.zip'):
            open_func = zipfile.Zipfile
            kwargs = {}
        elif self.dev_conllu_file[0].endswith('.gz'):
            open_func = gzip.open
            kwargs = {}
        elif self.dev_conllu_file[0].endswith('.xz'):
            open_func = lzma.open
            kwargs = {'errors': 'ignore'}
        else:
            open_func = codecs.open
            kwargs = {'errors': 'ignore'}

        with open_func(self.test_conllu_file[0], 'rb') as f:
            reader = codecs.getreader('utf-8')(f, **kwargs)
            for line in reader:
                line = line.strip()
                if line and line.startswith('#'):
                    # if re.match('[0-9]+[-.][0-9]+', line):
                    self.id2flag[line] = 0
                    self.index2flag[num] = 0
                    num += 1
        return num
    # =============================================================
    #--------------------test---------------------------------------
    
    

    # =============================================================
    def dump(self):
        """"""
        with codecs.open(self.flag_savename, 'w', encoding='utf-8', errors='ignore') as f:
            for id in self.id2flag:
                f.write(u'{}\t{}\n'.format(id, self.id2flag[id]))
        return
    # =============================================================
    
    # =============================================================
    def split(self):
        
        if self.train_conllu_file[0].endswith('.zip'):
            open_func = zipfile.Zipfile
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.gz'):
            open_func = gzip.open
            kwargs = {}
        elif self.train_conllu_file[0].endswith('.xz'):
            open_func = lzma.open
            kwargs = {'errors': 'ignore'}
        else:
            open_func = codecs.open
            kwargs = {'errors': 'ignore'}

        su_ids = [ids for ids, flag in self.id2flag.items() if flag == 1]


        su_file = self.train_conllu_file[0] + '.portion%d.su'%(self.portion)
        w = codecs.open(su_file, 'w', encoding='utf-8', errors='ignore')

        buff = []

        with open_func(self.train_conllu_file[0], 'rb') as f:
            reader = codecs.getreader('utf-8')(f, **kwargs)
            for line in reader:
                buff.append(line)
                if line.startswith('#'):
                    current = line
                if line == '\n':
                    if current.strip() in su_ids:
                        for contents in buff:
                            w.write(contents)
                    buff = []
        w.close()

        return
    # =============================================================


    @property
    def save_dir(self):
        return self._config.getstr(self, 'save_dir')
    @property
    def train_conllu_file(self):
        return self._config.getfiles(self, 'train_conllus')

    @property
    def dev_conllu_file(self):
        return self._config.getfiles(self, 'dev_conllus')

    @property
    def test_conllu_file(self):
        return self._config.getfiles(self, 'test_conllus')

    @property
    def flag_savename(self):
        return os.path.join(self.save_dir, 'flag' + '.lst')

    @property
    def classname(self):
        return self.__class__.__name__
    
    @property
    def portion(self):
        return self._config.getint(self, 'portion')
    
    @property
    def fix(self):
        return self._config.getboolean(self, 'fix_label_data')
    
    @property
    def labeled_num(self):
        return self._config.getint(self, 'labeled_num')
