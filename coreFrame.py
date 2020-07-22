#mian framework class and dataset reader
import random

import pandas as pd
from for_null import process_null
from model_base.basicModel import getModel
from evaluationFunction import basic_function

class classifierFramework:
    def __init__(self,name='',comment=''):
        self.has_train_set = False
        self.has_test_set = False
        self.has_model_type = False
        self.has_fitted_model = False
        self.has_fearture_select_strategy = False
        self.has_evaluation_function = False
        self.train_test_featrue_match =False
        self.test_result_done = False
        self.name = name
        self.comment = comment
        self.eva_function_list = []
        self.predict_score = []
        self.predict_result = []
        self.evaluation_dict = {}
    
    def reader(self,file_path,splitType='\t',label='label',feature_list='',test_train_type='random',random_ration=1,train_balance=False,for_null='delete',save_set=False):
        dataset = pd.read_csv(file_path,sep=splitType)  #read
        self.labelname = label
        if feature_list != '':                          #new mian_test_set ,process null cell
            feature_list.append(label)
            if test_train_type== 'name':
                feature_list.append('set_name')
            self.main_dataset = self.set_null(dataset.loc[:, feature_list],for_null)
        else:
            self.main_dataset = self.set_null(dataset,for_null)
        #self.dataset_len = 1000
        #self.positive_set_len = 300
        #self.negative_set_len = 600 #for test
        self.dataset_len = self.main_dataset.shape[0]
        self.positive_set_len = self.main_dataset[self.main_dataset[label].isin(['1'])].shape[0]
        self.negative_set_len = self.main_dataset[self.main_dataset[label].isin(['0'])].shape[0]
        if test_train_type== 'name':
            self.train_set = self.main_dataset[self.main_dataset['set_name'].isin(['train'])]
            self.test_set = self.main_dataset[self.main_dataset['set_name'].isin(['test'])]
        elif test_train_type== 'random':
            if self.positive_set_len<= self.negative_set_len:
                self.test_positive_draw,self.train_positive_draw,self.test_negative_draw,self.train_negative_draw = self.get_draw(self.dataset_len,self.positive_set_len,self.negative_set_len,random_ration,train_balance)
            else:
                self.test_negative_draw,self.train_negative_draw,self.test_positive_draw,self.train_positive_draw = self.get_draw(self.dataset_len,self.negative_set_len,self.positive_set_len,random_ration,train_balance)
            self.train_set,self.test_set = self.__get_random_tt_set()
        self.has_train_set =True
        self.has_test_set = True
        self.train_test_featrue_match = True
        if save_set:
            if self.name == '':
                file_head = 'work' + '_' + str(random.randint(0,999999)) + '_'
            else:
                file_head = 'work' + '_' + self.name + '_'
            self.main_dataset.to_csv('workDataset/' + file_head + 'full_set.csv',index=0)
            self.train_set.to_csv('workDataset/' + file_head + 'train_set.csv',index=0)
            self.test_set.to_csv('workDataset/' + file_head + 'test_set.csv',index=0)

        #self.trainX,self.train_label,self.testX,self.test_label = self.__get_split_tt()
        self.__get_split_tt() #get self.trainX,self.train_label,self.testX,self.test_label

        #print(dataset)
    def train_set_reader(self,file_path,splitType='\t',label='label',feature_list='',for_null='delete'):
        return 0
        #last if has test set ,match them
    def test_set_reader(self,file_path,splitType='\t',label='label',feature_list='',for_null='delete'):
        #last if has train set ,match them
        return 0
    #add null process here   
    def set_null(self,df,for_null): 
        if for_null == 'delete':
            df = process_null.delete_null(df)
        return df
            
    def __get_split_tt(self):
        col_name =list(self.train_set)
        new_col_name = []
        delete_list =['set_name','random_set',self.labelname]
        for i in col_name:
            if i not in delete_list:
                new_col_name.append(i)

        self.train_label = self.train_set[self.labelname]
        self.trainX = self.train_set[new_col_name]

        self.test_label = self.test_set[self.labelname]
        self.testX = self.test_set[new_col_name]
        self.feature = new_col_name




    def __get_random_tt_set(self):
        positive_set = self.main_dataset[self.main_dataset['label'].isin(['1'])]
        negative_set = self.main_dataset[self.main_dataset['label'].isin(['0'])]

        train_pos = positive_set.sample(n=self.train_positive_draw,axis=0)
        test_pos = positive_set.sample(n=self.test_positive_draw,axis=0)

        train_neg = negative_set.sample(n=self.train_negative_draw,axis=0)
        test_neg = negative_set.sample(n=self.test_negative_draw,axis=0)

        train_set= train_pos.append(train_neg,ignore_index=True)
        train_set['random_set'] = 'train'

        test_set = test_pos.append(test_neg,ignore_index=True)
        test_set['random_set'] = 'test'
        return train_set,test_set


    def get_draw(self,total,less,more,random_ration,train_balance):
        if not train_balance:
            more_train_draw = int((random_ration/(random_ration+1)) * more)
            more_test_draw = more - more_train_draw
            less_train_draw = int((random_ration/(random_ration+1)) * less)
            less_test_draw = less - less_train_draw
        else:
            mini_count = (random_ration / (random_ration + 1)) / 3 * total
            if less < mini_count:
                less_test_draw = int(less * 0.1 + 1)
                less_train_draw = less - less_test_draw
            elif less < 2 * mini_count:
                less_test_draw = int(less * 0.15 + 1)
                if less - less_test_draw >= int((random_ration / (random_ration + 1)) / 2 * total):
                    less_train_draw = int((random_ration / (random_ration + 1)) / 2 * total)
                    less_test_draw = less - less_train_draw
                else:
                    less_train_draw = less - less_test_draw
            else:
                less_train_draw = int((random_ration / (random_ration + 1)) / 2 * total)
                less_test_draw = less - less_train_draw                    
            more_train_draw = int((random_ration / (random_ration + 1)) * total - less_train_draw)
            more_test_draw = total -  less_train_draw - less_test_draw - more_train_draw

        return less_test_draw,  less_train_draw,  more_test_draw, more_train_draw 
    

    def setModel(self,model_type):

        if model_type.lower() == 'randomforest':
            self.model,self.model_info = getModel('randomforest') 
            self.model_type = 'randomforest'

            try:
                print(self.model_info['summary'])
                self.has_model_type = True
            except Exception as e:
                print('####################### set model error! #######################')
                print(e)




        return self.model


    def setEvaluationFunction(self,funcType='acc'):
        
        if len(funcType[0]) == 1:  #if funcType is a string
           funcType = [funcType] 
        for func in funcType:
            if func.lower() == 'acc':
                self.eva_function_list.append('acc')
                self.has_evaluation_function = True
            elif func.lower() == 'mcc':
                self.eva_function_list.append('mcc')
                self.has_evaluation_function = True
            elif func.lower() == 'auc':
                self.eva_function_list.append('auc')
                self.has_evaluation_function = True

    def fitModel(self,feature_list = ''):
        if self.has_train_set and self.has_model_type:
            print('## start fit the model! ##')
            
            #print(self.fitted_model.feature_importances_)
            self.feature_importances = {}
            #random forest sklearn
            ##############
            if self.model_type in ['randomforest']: #has importance model list /sklearn model
                self.fitted_model = self.model.fit(self.trainX,self.train_label)           
                for i in range(len(self.feature)):
                    self.feature_importances[self.feature[i]] = self.fitted_model.feature_importances_[i]
                # fit test set
                if self.has_test_set:
                    self.predict_result = self.model.predict(self.testX)
                    rr = self.model.predict_proba(self.testX)
                    for pred in rr:
                        self.predict_score.append(pred[1])
                if self.has_test_set and self.has_evaluation_function:
                    self.__get_evaluation_dict()

            ##############
        elif not self.has_train_set:
            print('## error : no train set! ##')
        else:
            print('## error : no model!')

    
    def __get_evaluation_dict(self):
        for func in self.eva_function_list:
            if func in ['acc','mcc','auc']:           # list of eva func
                result = self.__eva_func_compute(func,self.test_label,self.predict_result,self.predict_score)
                self.evaluation_dict[func] = result


    def __eva_func_compute(self,func,real_label,pred_label,pred_score):
        if func == 'acc':
            result = basic_function.acc_score(real_label,pred_label)
        if func == 'mcc':
            result = basic_function.mcc_score(real_label,pred_label)

        return result
    def __get_evaluation_dict_nosave(self,real_label,pred_label,pred_score):
        evaluation_dict ={}
        for func in self.eva_function_list:
            if func in ['acc','mcc','auc']:           # list of eva func
                result = self.__eva_func_compute(func,real_label,pred_label,pred_score)
                evaluation_dict[func] = result
        return evaluation_dict   
    def testModel(self,testX,test_label,feature_list = ''):
        print('## start fit the model! ##')
        #random forest sklearn
            ##############
        if self.model_type in ['randomforest']: #has importance model list /sklearn model
            #self.fitted_model = self.model.fit(self.trainX,self.train_label)           
            #for i in range(len(self.feature)):
                #self.feature_importances[self.feature[i]] = self.fitted_model.feature_importances_[i]
            # fit test set
            predict_score = []
            evaluation_dict = {}
               
            predict_result = self.model.predict(testX)
            rr = self.model.predict_proba(testX)
            for pred in rr:
                predict_score.append(pred[1])
            if self.has_evaluation_function:
                evaluation_dict = self.__get_evaluation_dict_nosave(test_label,predict_result,predict_score)
        return predict_result,predict_score,evaluation_dict

            ##############



def main():
    print('start work!')
    #a = experimentFramework()
if __name__ == '__main__':
    main()