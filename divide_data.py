import json
import random


min_log = 15


def divide_data():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    '''
    with open('data/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        stu_i += 1
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, train_set, val_set, test_set = [], [], [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_val = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.7)
        val_size = int(stu['log_num'] * 0.1)
        test_size = stu['log_num'] - train_size - val_size
        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_val['log_num'] = val_size
        stu_val['logs'] = logs[train_size:train_size+val_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        val_set.append(stu_val)
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('data/train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open('data/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('data/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set
    with open('data/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


def group_trainset_by_exercise():
    '''
    group logs in train_set by exercise to facilitate the computation of DOA
    :return:
    '''
    exers = FastRBTree()
    with open('data/train_set.json', encoding='utf8') as i_f:
        data = json.load(i_f)
    for log in data:
        exer_id = log['exer_id']
        exer = exers.get(exer_id)
        if exer is None:
            exer = {'exer_id': exer_id, 'knowledge_code': log['knowledge_code']}
            exer_log = [{'user_id': log['user_id'], 'score': log['score']}]
            exer['logs'] = exer_log
            exers.insert(exer_id, exer)
        else:
            exer['logs'].append({'user_id': log['user_id'], 'score': log['score']})
    json_format = []
    for exer_id in exers:
        json_format.append(exers.get(exer_id))

    with open('data/group_trainset_by_exercise.json', 'w', encoding='utf8') as o_f:
        json.dump(json_format, o_f, indent=4, ensure_ascii=False)


def group_trainset_by_knowledge():
    with open('data/group_trainset_by_exercise.json', encoding='utf8') as i_f:
        exers = json.load(i_f)
    exers_rbt = FastRBTree()
    for exer in exers:
        exers_rbt.insert(exer['exer_id'], exer)
    kn_rbt = FastRBTree()
    for exer in exers:
        for kn in exer['knowledge_code']:
            kn_value = kn_rbt.get(kn)
            if kn_value is None:
                kn_value = [{'exer_id': exer['exer_id'], 'logs': exer['logs']}]
                kn_rbt.insert(kn, kn_value)
            else:
                kn_value.append({'exer_id': exer['exer_id'], 'logs': exer['logs']})
    json_format = []
    for kn in kn_rbt:
        json_format.append({'knowledge_code': kn, 'exercises': kn_rbt.get(kn)})
    with open('data/group_trainset_by_knowledge.json', 'w', encoding='utf8') as o_f:
        json.dump(json_format, o_f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    divide_data()
    # group_trainset_by_exercise()
    # group_trainset_by_knowledge()
