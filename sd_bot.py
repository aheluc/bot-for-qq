from data import Dictionary
import time
import random
import pickle
import jieba
from gen import Replier
import os


MSG = {
    'INTEGER_ERROR': ['integer, please.my ADMIRAL.',
                      'integerを入力してにゃあ～' 
                     ],
    'UNKNOWN_COMMANDS': ['不懂的说～',
                         '不知道，问凑凑去',
                        ],
    'BELOW_ZERO': ['0以上の整数を入力してにゃあ～',
                  ],
    'FLOAT_ERROR': ['float, please.',
                   'floatを入力してにゃあ～' 
                  ],
    'RANGE_ERROR': ['0.1 - 1, please.',
                   '0.1 - 1で設定してにゃあ～' 
                  ],
}

ROOT = 'C:\\Users\\lenovo\\.qqbot-tmp\\plugins\\'

import configparser
config = configparser.ConfigParser()
config.read(ROOT + 'app.conf', encoding='utf8')
dict_path = config.get('DICTIONARY', 'dict_path')
model_save_path = config.get('MODEL', 'save_path')
embedding_dim = int(config.get('MODEL', 'embedding_dim'))
hidden_dim = int(config.get('MODEL', 'hidden_dim'))
num_layers = int(config.get('MODEL', 'num_layers'))

jieba.load_userdict(dict_path)

_dict = Dictionary([])
model_dict_path = os.path.join(model_save_path, 'freq1.dict')
_dict.load(model_dict_path)

rep = Replier(model_save_path, _dict, num_layers, embedding_dim, hidden_dim)
rep.load('freq1_(3.562772035598755)')

REPLY_TIME = {}
REPLY_TIME['ME'] = 0

def get_errmsg(key):
    limit = len(MSG[key]) - 1
    idx = random.randint(0, limit)
    return MSG[key][idx]

def onQQMessage(bot, contact, member, content):
    if content.startswith('[@ME]'):
        now = time.time()
        cooldown = now - REPLY_TIME['ME']
        REPLY_TIME['ME'] = now
        if cooldown < 7:
            cooldown = str(7 - cooldown)[0]
            bot.SendTo(contact, '少女冷却中...剩余 %s 秒.' % cooldown)
            return
        REPLY_TIME['ME'] = now
        content = content[5:]
        args = [arg for arg in content.split(' ') if arg]
        reply_msg = ''
        if len(args) == 0:
            bot.SendTo(contact, '没事你@个鸡儿。')
            return
        # roll
        elif args[0] == 'roll':
            if len(args) == 1:
                reply_msg += str(random.randint(1, 100))
            elif len(args) > 1:
                try:
                    _max = int(args[1])
                    if _max <= 0:
                        reply_msg += get_errmsg('BELOW_ZERO')
                    else:
                        reply_msg += str(random.randint(1, _max))
                except Exception as e:
                    reply_msg += get_errmsg('INTEGER_ERROR')
        # set temperature
        elif args[0] == 'set_temp':
            if len(args) == 1:
                rep.temperature = 0.6
                bot.SendTo(contact, 'temp reseted')
            elif len(args) > 1:
                try:
                    factor = float(args[1])
                    if 1 >= factor >= 0.1:
                        rep.temperature = factor
                    else:
                        reply_msg += get_errmsg('RANGE_ERROR')
                except Exception as e:
                    reply_msg += get_errmsg('FLOAT_ERROR')
        else:
            word = list(jieba.cut(content))
            word = rep.make_sentence(word)
            if word:
                reply_msg = ''.join(word)
            else:
                reply_msg += get_errmsg('UNKNOWN_COMMANDS')
        bot.SendTo(contact, reply_msg)
