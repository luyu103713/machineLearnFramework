from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef


def set_basic(func_name):
	return 0

def acc_score(Real,Pred):
	return accuracy_score(Real,Pred)

def mcc_score(Real,Pred):
	return  matthews_corrcoef(Real,Pred)