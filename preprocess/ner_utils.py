#coding:utf-8
###################################################
# File Name: ner_utils.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年10月28日 星期一 15时28分03秒
#=============================================================
def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for token, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": token, "start": idx, "end": idx + 1, "type": tag[2: ]})
        elif tag[0] == "B":
            entity_name += token
            entity_start = idx
        elif tag[0] == "I" or tag[0] == 'M':
            entity_name += token
        elif tag[0] == "E":
            entity_name += token
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2: ]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def bert_result_to_json(tokens, tags):
    item = {"string": ''.join(tokens), "entities": []}
    entity_name = ""
    entity_start = 0

    idx = 0
    flag = False
    for token, tag in zip(tokens, tags):
        if tag[0] == "S":
            item["entities"].append({"word": token, "start": idx, "end": idx + 1, "type": tag[2: ]})
        elif tag[0] == "B":
            entity_name += token
            entity_start = idx
            flag = True
        elif tag[0] == "I" or tag[0] == 'M':
            entity_name += token
        elif tag[0] == "E" and flag:
            entity_name += token
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2: ]})
            entity_name = ""
            flag = False
        else:
            entity_name = ""
            entity_start = idx
            flag = False
        idx += 1               
    return item    

if __name__ == '__main__':
    #tags = ['B-PERSON', 'E-PERSON', 'M-PERSON', 'E-PERSON',]
    #tags = ['B-PERSON', 'M-PERSON', 'E-PERSON', 'E-PERSON',]
    tags = ['B-PERSON', 'M-PERSON', 'O', 'E-PERSON',]

    tokens = ['欧', '阳', '明', '星']


    

    res = bert_result_to_json(tokens, tags)
    print(res)
