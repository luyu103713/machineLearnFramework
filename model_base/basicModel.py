
from sklearn.ensemble import RandomForestClassifier



def getModel(model_type):

    if model_type == 'randomforest':
        model= RandomForestClassifier(n_estimators = 500,n_jobs=-1)
        info_of_model = {'summary':'This RandomForestClassifier model base on sklearn.ensemble. ','parameter':{'n_estimators':500,'n_jobs':-1},'fitted':False}
    else:
        model = ''
        info_of_model = {}
    return model,info_of_model




def main():
    print('start work!')

if __name__ == '__main__':
    main()