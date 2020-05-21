# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
import tqdm
import json
import sys
import logging
import reader.task_reader as task_reader

# 权重初始化，默认xavier
def init_network(model, method='kaiming', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def calculate_acc(logits, labels, input_mask):
    # the golden metric should be "f1" in spo level
    # but here only "accuracy" is computed during training for simplicity (provide crude view of your training status)
    # accuracy is dependent on the tagging strategy
    # for each token, the prediction is counted as correct if all its 100 labels were correctly predicted
    # for each example, the prediction is counted as correct if all its token were correctly predicted
    #logits_lod = logits.lod()
    #labels_lod = labels.lod()
    #logits_tensor = np.array(logits)
    #labels_tensor = np.array(labels)
    #assert logits_lod == labels_lod

    num_total = 0
    num_correct = 0
    token_total = 0
    token_correct = 0
    for i  in range(len(logits)):
        output = logits[i]
        label = labels[i]
        mask = input_mask[i]
        if (mask==0):
            continue
        inference = output
        inference[output>=0.5]=1
        inference[output<0.5]=0
        if (inference==label).all():
            num_correct += 1
        num_total+=1

    return num_correct, num_total

def train(config, model):
    reader = task_reader.RelationExtractionMultiCLSReader(
        vocab_path="ERNIE_pretrain/vocab.txt",
        label_map_config='data/relation2label.json',
        spo_label_map_config = 'data/label2relation.json',
        max_seq_len=256,
        do_lower_case=True,
        in_tokens=False,
        random_seed=1,
        task_id=0,
        num_labels=112)
    train_iter = reader.data_generator(
        input_file = 'data/train_data.json',
        batch_size=16,
        epoch=20,
        shuffle=True,
        phase="train"
        )
    test_iter = reader.data_generator(
        input_file = 'data/dev_data.json',
        batch_size=16,
        epoch=1,
        shuffle=False,
        phase='test')

    output, f1 = evaluate(test_iter,model)
    return 
    num_train_examples = reader.get_num_examples('data/train_data.json')
    train_steps = 20*num_train_examples//16
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=train_steps) # epoch included
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    logger = logging.getLogger(__name__)
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    best_f1 = 0
    if True:


        for i,it in enumerate(train_iter()):
            sys.stdout.flush()

            sent = torch.squeeze(torch.tensor(it[0])).cuda()
            mask = torch.squeeze(torch.tensor(it[4])).cuda()
            label = torch.squeeze(torch.tensor(it[5])).cuda()
            lens = torch.squeeze(torch.tensor(it[6])).cuda()
            outputs = model(sent,mask)

         
            logits = torch.flatten(outputs,start_dim=0,end_dim=1)
            labels = torch.flatten(label,start_dim=0,end_dim=1)
            input_mask = torch.flatten(mask,start_dim=0,end_dim=1)
            log_logits = torch.log(logits)
            log_logits_neg = torch.log(1-logits)
            ce_loss = 0. - labels*log_logits-(1-labels)*log_logits_neg

            ce_loss = torch.mean(ce_loss)
            ce_loss = ce_loss*input_mask
            loss = torch.mean(x=ce_loss)

            lod = logits.cpu().detach().numpy()
            lab = labels.cpu().detach().numpy()
            msk = input_mask.cpu().numpy().tolist()
            num_correct, num_total = calculate_acc(lod,lab,msk)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            if (i>0 and i%10700==0):
                dev_iter = reader.data_generator(
                input_file = 'data/dev_data.json',
            batch_size=16,
            epoch=1,
            shuffle=False,
            phase="train"
                )

                output, f1 = evaluate(dev_iter,model)


                if (f1>best_f1):
                    torch.save(model.state_dict(), config.save_path)
                    best_f1 = f1








def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def post_process(inference,length):
    # this post process only brings limited improvements (less than 0.5 f1) in order to keep simplicity
    # to obtain better results, CRF is recommended
    reference = []
    for i,token in enumerate(inference):
        if (i>=length):
            break
        token_ = token.copy()
        token_[token_ >= 0.5] = 1
        token_[token_ < 0.5] = 0
        reference.append(np.argwhere(token_ == 1))

    #  token was classified into conflict situation (both 'I' and 'B' tag)
    for i, token in enumerate(reference[:-1]):
        if [0] in token and len(token) >= 2:
            if [1] in reference[i + 1]:
                inference[i][0] = 0
            else:
                inference[i][2:] = 0

    #  token wasn't assigned any cls ('B', 'I', 'O' tag all zero)
    for i, token in enumerate(reference[:-1]):
        if len(token) == 0:
            if [1] in reference[i - 1] and [1] in reference[i + 1]:
                inference[i][1] = 1
            elif [1] in reference[i + 1]:
                inference[i][np.argmax(inference[i, 1:]) + 1] = 1 

    #  handle with empty spo: to be implemented

    return inference

def evaluate(dev_iter,model):
    spo_label_map = json.load(open("data/label2relation.json"))


    examples = []
    model.eval()

    f = open("data/dev_data.json", 'r', encoding="utf8")
    for line in f.readlines():
        examples.append(json.loads(line))
    f.close()
    tp, fp, fn = 0, 0, 0

    output_file = open("test_out.txt",'w')
    with torch.no_grad():
        for index, it in enumerate(dev_iter()):
            sys.stdout.flush()

            start = time.clock()
            # prepare fetched batch data: unlod etc.
         #   [logits, labels,example_index_list, tok_to_orig_start_index_list, tok_to_orig_end_index_list ] = it
            example_index_list = it[7]
            tok_to_orig_start_index_list = it[8]
            tok_to_orig_end_index_list = it[9]
            example_index_list = np.array(example_index_list).astype(
                int) - 100000

            sent = torch.squeeze(torch.tensor(it[0])).cuda()
            mask = torch.squeeze(torch.tensor(it[4])).cuda()
            label = it[5]
            lens = it[6]
            outs = model(sent,mask)

            logits = outs.cpu().detach().numpy()

            #   no .lod() in torch, use mask/seq length instead

            for i in range(np.size(logits,0)):

                start2 = time.clock()
                # prepare prediction results for each example
                example_index = example_index_list[i]
                example = examples[example_index]
                tok_to_orig_start_index = tok_to_orig_start_index_list[i]
                tok_to_orig_end_index = tok_to_orig_end_index_list[i]
                inference_tmp = logits[i]
                labels_tmp = label[i]
                l = lens[i]
                # some simple post process
                inference_tmp = post_process(inference_tmp,l)


                # logits -> classification results
                inference_tmp[inference_tmp >= 0.5] = 1
                inference_tmp[inference_tmp < 0.5] = 0
                predict_result = []
                for j,token in enumerate(inference_tmp):
                    if (j>=l):
                        break
                    predict_result.append(np.argwhere(token == 1).tolist())
                # format prediction into spo, calculate metric


                formated_result = format_output(
                    example, predict_result, spo_label_map,
                    tok_to_orig_start_index, tok_to_orig_end_index,l)

                tp_tmp, fp_tmp, fn_tmp = calculate_metric(
                    example['spo_list'], formated_result['spo_list'],l)
               # formated_result['text']=example['text']
              #  fk = json.dump(formated_result,output_file,ensure_ascii=False)
               # print('\n',file=output_file,end='')

                tp += tp_tmp
                fp += fp_tmp
                fn += fn_tmp


    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f = 2 * p * r / (p + r) if p + r != 0 else 0
    return "[evaluation] precision: %f, recall: %f, f1: %f" % (
        p, r, f), f


def calculate_metric(spo_list_gt, spo_list_predict,length):
    # calculate golden metric precision, recall and f1
    # may be slightly different with final official evaluation on test set,
    # because more comprehensive detail is considered (e.g. alias)
    tp, fp, fn = 0, 0, 0
    for spo in spo_list_predict:
        flag = 0
        for spo_gt in spo_list_gt:
            if spo['predicate'] == spo_gt['predicate'] and spo[
                    'object'] == spo_gt['object'] and spo['subject'] == spo_gt[
                        'subject']:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1
    '''
    for spo in spo_list_predict:
        if spo in spo_list_gt:
            tp += 1
        else:
            fp += 1
            '''

    fn = len(spo_list_gt) - tp
    return tp, fp, fn

def format_output(example, predict_result, spo_label_map,
                  tok_to_orig_start_index, tok_to_orig_end_index,length):
    # format prediction into example-style output
    complex_relation_label = [8, 10, 26, 32, 46]
    complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]
    instance = {}
    predict_result = predict_result[1:len(predict_result) -
                                    1]  # remove [CLS] and [SEP]
    text_raw = example['text']

    flatten_predict = []
    for layer_1 in predict_result:
        for layer_2 in layer_1:
            flatten_predict.append(layer_2[0])

    subject_id_list = []
    for cls_label in list(set(flatten_predict)):
        if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predict:
            subject_id_list.append(cls_label)
    subject_id_list = list(set(subject_id_list))

    def find_entity(id_, predict_result):
        entity_list = []
        for i in range(len(predict_result)):
            if [id_] in predict_result[i]:
                j = 0
                while i + j + 1 < len(predict_result):
                    if [1] in predict_result[i + j + 1]:
                        j += 1
                    else:
                        break
                entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                          tok_to_orig_end_index[i + j] + 1])
                entity_list.append(entity)

        return list(set(entity_list))

    spo_list = []
    jj = time.clock()
  #  print(len(subject_id_list))
    for id_ in subject_id_list:
        if id_ in complex_relation_affi_label:
            continue
        if id_ not in complex_relation_label:
            ii = time.clock()
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
            #print("fine:",time.clock()-ii)
        #    print(len(subjects))
         #   print(len(objects))
            for subject_ in subjects:
                for object_ in objects:
                    spo_list.append({
                        "predicate": spo_label_map['predicate'][id_],
                        "object_type": {
                            '@value': spo_label_map['object_type'][id_]
                        },
                        'subject_type': spo_label_map['subject_type'][id_],
                        "object": {
                            '@value': object_
                        },
                        "subject": subject_
                    })
        else:
            ii = time.clock()
            #  traverse all complex relation and look through their corresponding affiliated objects
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
           # print("fine:",time.clock()-ii)
         #   print(len(subjects))
          #  print(len(objects))
            for subject_ in subjects:
                for object_ in objects:
                    kk = time.clock()
                    object_dict = {'@value': object_}
                    object_type_dict = {
                        '@value':
                        spo_label_map['object_type'][id_].split('_')[0]
                    }
                  #  print("1:",time.clock()-kk)

                    if id_ in [8, 10, 32, 46] and id_ + 1 in subject_id_list:
                        id_affi = id_ + 1
                        object_dict[spo_label_map['object_type'][id_affi].split(
                            '_')[1]] = find_entity(id_affi + 55,
                                                   predict_result)[0]
                        object_type_dict[spo_label_map['object_type'][
                            id_affi].split('_')[1]] = spo_label_map[
                                'object_type'][id_affi].split('_')[0]
                    elif id_ == 26:
                        for id_affi in [27, 28, 29]:
                            if id_affi in subject_id_list:
                                object_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                find_entity(id_affi + 55, predict_result)[0]
                                object_type_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                spo_label_map['object_type'][id_affi].split('_')[0]
                   # print("2:",time.clock()-kk)
                    spo_list.append({
                        "predicate": spo_label_map['predicate'][id_],
                        "object_type": object_type_dict,
                        "subject_type": spo_label_map['subject_type'][id_],
                        "object": object_dict,
                        "subject": subject_
                    })
   # print("tot",time.clock()-jj)
    instance['text'] = example['text']
    instance['spo_list'] = spo_list
    return instance
