import sys
import os
import json
import sklearn
a = open("data/dev_data.json",'r')
b = open("test_out.json","r")
aa = a.readlines()
bb = b.readlines()
answ = {}
answ_tot = {}

pred = {}
pred_tot = {}

for l,r in (zip(aa,bb)):
	ans = json.loads(l.strip())
	res = json.loads(r.strip())

	for spo in ans['spo_list']:
		if ( not spo['predicate'] in answ.keys()):
			answ[spo['predicate']]=0
			answ_tot[spo['predicate']]=1
		else:
			answ_tot[spo['predicate']]+=1
		r=0
		temp = None
		shot = [0]*len(res['spo_list'])
		for i,spo2 in enumerate(res['spo_list']):
			p = spo['predicate']
			if (p!=spo2['predicate']):
				continue
			if  ((spo2['object_type'] == spo['object_type']) and spo2['subject_type'] == spo['subject_type'] and spo['object']==spo2['object'] and spo['subject']==spo2['subject']):
				r=1
				answ[spo['predicate']]+=1
				shot[i]=1
				break



		for i,spo2 in enumerate(shot):
			if (not res['spo_list'][i]['predicate'] in pred.keys()):
				pred[res['spo_list'][i]['predicate']]=0
				pred_tot[res['spo_list'][i]['predicate']]=1
			else:
				pred_tot[res['spo_list'][i]['predicate']]+=1
			if (spo2==1):
				pred[res['spo_list'][i]['predicate']]+=1

print("--------recall----------")

for key in answ_tot.keys():
	print(key,":",answ[key],'/',answ_tot[key],'=',"%.2f"%(answ[key]*1.0/answ_tot[key]))
print("--------precision-------")
for key in pred.keys():
	print(key,":",pred[key],'/',pred_tot[key],'=',"%.2f"%(pred[key]*1.0/pred_tot[key]))





