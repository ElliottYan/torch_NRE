import pandas as pd
import pickle
# from pymongo import MongoClient
from collections import defaultdict
from structures import time_signature, Stack, Mention
import random, math
import pdb
import spacy
from spacy.symbols import ORTH, LEMMA, POS


nlp = spacy.load('en', disable=['ner'])
data_root = '/data/yanjianhao/nlp/NER/'
special_case = [{ORTH: u'<t>'},]
nlp.tokenizer.add_special_case(u'<t>', special_case)


def set_default():
    return None


def clean(row):
    if type(row) != str:
        #         print(row)
        return 'NaN'
    elif len(row) < 20:
        return 'NaN'
    else:
        return row[:10]


def check_relation(label, x):
    # label : a list of time_signature
    # x : a query for its label
    stack = Stack()
    unpop = []
    for node in label:
        if node < x:
            if node.type == 'start':
                stack.push(node)
                # print('push')
            if node.type == 'end':
                if node.relation == stack.peek().relation:
                    stack.pop()
                    # print('pop')
                    while(unpop and unpop[-1] == stack.peek().relation):
                        stack.pop()
                        # print('pop')
                        unpop = unpop[:-1]
                else:
                    unpop.append(node.relation)
        else:
            if node.type == 'end' and stack.peek().relation == 'NA' and node.relation != 'NA':
                stack.push(time_signature('0000-00-00', relation=node.relation))
            return stack.peek().relation


def create_labels():
    with open(data_root + "alignment.dat", 'rb') as f:
        align = pickle.load(f)

    entities_pair = pd.read_csv(data_root + "origin_data/entities.csv")

    formal_entities_pair = pd.concat([entities_pair[['entity1','entity2', 'entity1Label', 'entity2Label', 'relation_name']],
                                      entities_pair['start_time'].apply(clean) ,entities_pair['end_time'].apply(clean)], axis=1)

    # This is for creating label sequences
    labels = defaultdict(list)
    for ix, row in formal_entities_pair.iterrows():
        row = row
        en1 = row['entity1Label']
        en2 = row['entity2Label']
        rel = row['relation_name']
        if labels[(en2, en1)] is not None:
            # exchange en1 & en2
            en1, en2 = en2, en1

    #   initialization for labels
    #   each time signature denotes the end of some relation
        if not labels[(en1, en2)]:
            labels[(en1, en2)].append(time_signature('0000-00-00', relation='NA', node_type='start'))
            labels[(en1, en2)].append(time_signature('9999-99-99', relation='NA', node_type='end'))
        if row['start_time'] != 'NaN':
            labels[(en1, en2)].append(time_signature(row['start_time'], relation=rel, node_type='start'))
        if row['end_time'] != 'NaN':
            labels[(en1, en2)].append(time_signature(row['end_time'], relation=rel, node_type='end'))
    return labels
    # print(label for label in labels[:10])


def unit_test(labels):
    # This is unit test
    label = labels[('Euro', 'Estonia')]
    label.append(time_signature('2012-01-01', relation='test'))
    label.append(time_signature('2015-01-01', relation='currency', node_type='end'))
    label.sort()

    print([(t.time, t.relation, t.type) for t in labels[('Euro', 'Estonia')]])

    tmp1 = time_signature('_'.join(['2015', '01', '01']), relation='NA', node_type='mention')
    tmp2 = time_signature('_'.join(['2015', '11', '01']), relation='NA', node_type='mention')
    label = labels[('Euro', 'Estonia')]

    print(check_relation(label, tmp1))
    print(check_relation(label, tmp2))


def construct_dataset(file_path, labels, w_to_ix):
    with open(data_root + '/origin_data/mentions2/2018_01_24/mentions.csv', 'r') as f:
        lines = f.readlines()[1:]

    rel_to_ix = defaultdict(set_default)
    en_to_rel = defaultdict(set_default)
    mentions = defaultdict(list)

    # count = 0
    for line in lines:
        # count += 1
        # if count > 5:
        #     break
        line = line.split(',', maxsplit=9)
        # print(line)
        # extract all infos from train.txt
        rel, en1, en2, pos1, pos2 = line[1:6]
        year, month, day = line[6:9]
        sent = line[9].split()
        #     print(year, month, day)
        #   swap in case en1 and en2 's order may differ
        if labels[(en2, en1)]:
            en1, en2 = en2, en1
        tmp = time_signature('_'.join([year, month, day]), node_type='mention')
        tag = check_relation(labels[(en1, en2)], tmp)
        if rel_to_ix[tag] == None:
            rel_to_ix[tag] = len(rel_to_ix) - 1
        # turn tag into int
        tag = rel_to_ix[tag]

        # mentions[(en1, en2)].append()
        sent = [w_to_ix[word] if w_to_ix[word] else w_to_ix['UNK'] for word in sent]
        # mentions.append((pos1, pos2, sent, year, month, day, tag))
        mentions[(en1, en2)].append(Mention(sent, tag=tag, pos1=pos1, pos2=pos2, time=tmp))

    # keep mentions sorted
    for key, item in mentions.items():
        item.sort()

    # save intermediate results
    with open("/data/yanjianhao/nlp/torch/torch_NRE/origin_data/relaton_to_ix.txt", 'w') as f:
        lines = []
        for key, value in rel_to_ix.items():
            lines.append(str(key) + " " + str(value) + "\n")
        f.writelines(lines)

    with open("/data/yanjianhao/nlp/torch/torch_NRE/origin_data/mentions_train.dat", 'w') as fout:
        pickle.dump(mentions, fout, protocol=2)
    return mentions, rel_to_ix

def tokenization(sent):
    sent = nlp(sent)
    w = []
    for word in sent:
        w.append(word.text)
    return " ".join(w)



def separate_datasets(labels):
    # need separation from train to test

    with open(data_root + '/origin_data/mentions2/2018_01_24/mentions.csv', 'r') as f:
        lines = f.readlines()[1:]
    count = 0
    mentions_count = defaultdict(int)
    mentions = defaultdict(list)
    rel_2_ix = {}
    print('extracting mentions counts from csv file')
    for l in lines[1:]:
        # count += 1
        # if count > 20:
        #     break
        line = l.split(',', maxsplit=9)
        # extract all infos from train.txt
        rel, en1, en2, pos1, pos2 = line[1:6]
        year, month, day = line[6:9]
        sent = line[9]
        # extract all infos from train.txt

        if labels[(en2, en1)]:
            en1, en2 = en2, en1
        mentions_count[(en1, en2)] += 1
        l = tokenization(l)
        mentions[(en1, en2)].append(l)

    tmp = list(mentions.keys())
    # random.shuffle is in-place operation
    random.shuffle(tmp)
    train_rate = 0.7
    train = tmp[:math.floor(train_rate * len(tmp))]
    test = tmp[math.floor(train_rate * len(tmp)):]
    sum = 0
    for i in train:
        sum += mentions_count[i]
    print('There are %d mentions sentence in train!'%sum)

    train_sents = []
    test_sents = []
    for i in train:
        train_sents += mentions[i]
    for i in test:
        test_sents += mentions[i]
    with open("/data/yanjianhao/nlp/torch/torch_NRE/data/train_temporal.txt", 'w') as fout:
        fout.writelines(train_sents)
    with open("/data/yanjianhao/nlp/torch/torch_NRE/data/test_temporal.txt", 'w') as fout:
        fout.writelines(test_sents)


if __name__ == "__main__":
    labels = create_labels()
    separate_datasets(labels)

# unit_test()