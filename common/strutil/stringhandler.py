#coding:utf-8
###################################################
# File Name: stringhandler.py
# Author: Meng Zhao
# mail: @
# Created Time: Mon 26 Mar 2018 10:40:57 AM CST
#=============================================================

def normalize_num(word):
    if word.isdigit() and len(word) == 11:
        return '<phone>'
    elif word.isdigit():
        return '<num>'
    else:
        return word

def filter_sent(sent_str, stop_set):
    sent = sent_str.split(' ')
    new_sents = []
    for word in sent:
        new_word = normalize_num(word)
        if new_word not in stop_set:
            new_sents.append(new_word)
    new_sents_str = ' '.join(new_sents)
    return new_sents_str


def split_word_and_seg(trunks_str, stop_set):
    trunks = trunks_str.split(' ')
    words = []
    segs = []
    for trunk_str in trunks:
        trunk = trunk_str.rsplit('/', 1)
        if len(trunk) != 2:
            continue
        word, seg = trunk
        new_word = normalize_num(word)
        if new_word not in stop_set:
            words.append(new_word)
            segs.append(seg)
    sent = ' '.join(words)
    segs_str = ' '.join(segs)
    return sent, segs_str


if __name__ == '__main__':
    print(normalize_num('12345'))
    print(normalize_num('aa'))
    print(normalize_num('12a345'))
    sent, segs_str = split_word_and_seg('泛微/n 软件/n 大楼/n //a', set())
    print(sent)
    print(segs_str)
