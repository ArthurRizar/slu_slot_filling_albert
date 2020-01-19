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

def hard_bert_result_to_json(tokens, tags):
    item = {"string": ''.join(tokens), "entities": []}

    flag = False
    last_tag = 'O'
    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag[0] == "S":
            item["entities"].append({"word": token, "start": idx, "end": idx + 1, "type": tag[2: ]})
        elif tag[0] == "B":
            entity_start = idx
            entity_name = token
            flag = True
        elif (tag[0] == "I" or tag[0] == 'M') and flag:
            entity_name += token
        elif tag[0] == "E" and flag:
            entity_name += token
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2: ]})
            flag = False
        else:
            flag = False
        last_tag = tag
    return item

def bert_result_to_json(tokens, tags):
    item = {"string": ''.join(tokens), "entities": []}

    flag = False
    last_tag = 'O'
    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag[0] == "S":
            item["entities"].append({"word": token, "start": idx, "end": idx + 1, "type": tag[2: ]})
        elif tag[0] == "B":
            if last_tag[0] != 'B':
                entity_start = idx
                entity_name = token
            else:
                entity_name += token
            flag = True
        elif (tag[0] == "I" or tag[0] == 'M') and flag:
            entity_name += token
        elif tag[0] == "E" and flag:
            entity_name += token
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2: ]})
            flag = False
        else:
            flag = False
        last_tag = tag
    return item    

if __name__ == '__main__':
    #tags = ['B-PERSON', 'E-PERSON', 'M-PERSON', 'E-PERSON',]
    #tags = ['B-PERSON', 'M-PERSON', 'E-PERSON', 'E-PERSON',]
    #tags = ['B-PERSON', 'M-PERSON', 'O', 'E-PERSON',]
    tags = ['B-PERSON', 'M-PERSON', 'M-PERSON', 'E-PERSON',]


    tokens = ['欧', '阳', '明', '星']

    tags = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PERSON', 'B-PERSON', 'E-PERSON']
    #tags = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PERSON', 'O', 'B-PERSON', 'E-PERSON']
    #tags = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'M-PERSON', 'O', 'B-PERSON', 'E-PERSON']
    tokens =['把', '我', '明', '天', '的', '请', '假', '流', '程', '代', '理', '给', '聂', '李' ,'兵']
    

    res = bert_result_to_json(tokens, tags)
    #res = hard_bert_result_to_json(tokens, tags)
    print(res)
