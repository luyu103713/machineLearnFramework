from coreFrame import classifierFramework

a = classifierFramework(name='test01',comment='123')
random_ration = 1
a.reader('test01.csv',splitType=',',label='label',test_train_type='random',for_null='delete',random_ration=random_ration,train_balance=True,save_set=True)
print(a.dataset_len,a.positive_set_len)

print(a.test_positive_draw,a.train_positive_draw,a.test_negative_draw,a.train_negative_draw)
print('#############')

print('set_total : ',a.dataset_len)
print('set_positive : ',a.positive_set_len)
print('set_negative : ',a.negative_set_len)
print('train:test : ',random_ration)
print('test_positive_draw : ',a.test_positive_draw)
print('train_positive_draw : ',a.train_positive_draw)
print('test_negative_draw : ',a.test_negative_draw)
print('train_negative_draw : ',a.train_negative_draw)

print(a.trainX)
#print(a.test_set)
a.setModel('randomforest')

#print(type(p))

print(a.model.n_estimators)
a.setEvaluationFunction(['acc','mcc'])
a.fitModel()
print(a.predict_score)
print('####################')
print(a.evaluation_dict)
print('####################')

predict_result,predict_score,evaluation_dict = a.testModel(a.testX,a.test_label)

print(evaluation_dict)