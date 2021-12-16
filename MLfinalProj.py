
import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")




class MLproj:
    
    def __init__(self,  X:pd.DataFrame, 
                        y:pd.Series, 
                        start_numratio = 0.75, 
                        pred_num = 200 
                        ) -> None:
        """
        X: input varibables
        y: target values corresponding to X
        """
        self.X = X
        self.y = y        
        self.start = int(len(self.X) * start_numratio) # starting number 
        self.pred_num = pred_num # number of prediction returned by each ml model
        self.pred_array = np.zeros(pred_num) # prediction array from each ml model
        self.pnl = np.zeros(pred_num) #portfolio net value curve



    def predictor_SVM(self, Kernel, C):
        """
        prediction by support vector machine
        """
        X = self.X.drop("close", axis=1)
        Columns = X.columns
        X = scale(X)
        X = pd.DataFrame(X, columns=Columns)
        
        svc = svm.SVC(C = C, kernel=Kernel )
        for i in range(self.start, self.start + self.pred_num):
            Xtrain = X.iloc[:i, :]
            ytrain = self.y.iloc[:i]
            svc.fit(Xtrain, ytrain)
            sample = pd.DataFrame(X.iloc[i, :]).T #to be predicted 
            pred_val = svc.predict(sample)[0] #predict next direction -1 or +1
            self.pred_array[i - self.start] = pred_val # populate predarray by pred value

        return self.pred_array
        #return None


    # ==================== yuhao add ====================
    def feature_select(self, X_train, y_train, X_test) :
        """
        select by Extra Tree Feature Importance
        """
        ET = ExtraTreesClassifier()
        ET = ET.fit(X_train, y_train)
        select = SelectFromModel(ET, prefit=True)
        X_train = select.transform(X_train)
        X_test = select.transform(X_test)
        return X_train, X_test
        

    def predictor_GBDT(self) :
        """
        prediction by GBDT
        """
        X = self.X.copy()
        gbdt_tree = GradientBoostingClassifier()
        for i in range(self.start, self.start + self.pred_num):
            X_train = X.iloc[:i, :]
            y_train = self.y.iloc[:i]
            X_test = pd.DataFrame(X.iloc[i, :]).T
            X_train, X_test = self.feature_select(X_train, y_train, X_test)
            gbdt_tree.fit(X_train, y_train)
            pred_val = gbdt_tree.predict(X_test)[0]
            self.pred_array[i - self.start] = pred_val
        
        return self.pred_array
    
    def predictor_Random(self):
        """
        prediction by Random Walk
        """
        random_array = np.random.binomial(1, 0.5, self.pred_num)
        self.pred_array = np.array([-(-1)**i for i in random_array])
        return self.pred_array
    # ========================= =========================

    def predictor_Naive(self, freq):
        """
        prediction by Naive buy and sell every frequence day
        """
        buystate = True
        for i in range(self.pred_num):
            if (i % freq) == 0 and buystate:
                self.pred_array[i] = 1
                buystate = False
            elif (i % freq) == 0 and not buystate:
                self.pred_array[i] = -1
                buystate = True
            else:
                self.pred_array[i] = 0
        return self.pred_array

    def predictor_Logistics(self,c=1,solver='lbfgs'):
        """
        prediction by logistic regression
        """
        X = scale(self.X)
        X = pd.DataFrame(X, columns=self.X.columns)
        LogisticsReg = LogisticRegression(C=c, solver=solver)
        for i in range(self.start, self.start + self.pred_num):
            Xtrain = X.iloc[:i, :]
            ytrain = self.y.iloc[:i]
            LogisticsReg.fit(Xtrain, ytrain)
            sample = pd.DataFrame(X.iloc[i, :]).T #to be predicted
            pred_val = LogisticsReg.predict(sample)[0] #predict next direction -1 or +1
            self.pred_array[i - self.start] = pred_val # populate predarray by pred value

        return self.pred_array

    def predictor_NeuralNetwork(self,solver,hidden_layer_size):
        """
                prediction by logistic regression
                """
        X = self.X
        #X = scale(self.X)
        #X = pd.DataFrame(X, columns=self.X.columns)
        clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_size, random_state=1)

        for i in range(self.start, self.start + self.pred_num):
            Xtrain = X.iloc[:i, :]
            ytrain = self.y.iloc[:i]
            clf.fit(Xtrain, ytrain)
            sample = pd.DataFrame(X.iloc[i, :]).T  # to be predicted
            pred_val = clf.predict(sample)[0]  # predict next direction -1 or +1
            self.pred_array[i - self.start] = pred_val  # populate predarray by pred value

        return self.pred_array


    def backtestor_Pnlcurve(self):
        start = self.start 
        end = self.start + self.pred_num
        closeprice_change = self.X['close'].pct_change().shift(-1).iloc[start : end]
        pnl = (closeprice_change * self.pred_array + 1)
        pnl.iloc[0] = 1
        self.pnl = pnl.cumprod()
        plt.plot(list(range(self.pred_num)), self.pnl)
        plt.show()

        return None

    def backtestor_Pnlcurves(self,pre,name):
        start = self.start
        end = self.start + self.pred_num
        closeprice_change = self.X['close'].pct_change().shift(-1).iloc[start : end]
        plt.figure(figsize=(10, 6))

        for i in range(len(pre)):
            pnl = (closeprice_change * np.array(pre[i]) + 1)
            pnl.iloc[0] = 1
            pnl = pnl.cumprod()
            plt.plot(list(range(self.pred_num)), pnl,label = name[i])
        plt.xlabel('observations')
        plt.ylabel('P/L')
        plt.title('P/L curves for different machine learning methods')
        plt.legend()
        plt.show()

        return None


    def backtestor_accuracy(self):
        start = self.start 
        end = self.start + self.pred_num
        y_actual = self.y.iloc[start : end].values
        #acc = (np.abs(self.pred_array - y_actual) < 0.00001).sum()/len(y_actual)
        acc = accuracy_score(y_actual,self.pred_array)

        return acc



    def backtestor_decay(self,pre,n, name, Xlabel, Ylabel, long=None):
        start = self.start
        end = self.start + self.pred_num + n
        closeprice_change = self.X['close'].pct_change().shift(-1).iloc[start : end]
        res = []
        # flag = 0
        for i in range(len(pre)):
            # if pre[i]==flag:
            #     continue
            # flag = pre[i]
            if long is not None:
                if long:
                    if pre[i]==-1:
                        continue
                else:
                    if pre[i]==1:
                        continue
            temp = [1]
            for j in range(n):
                temp.append(temp[-1]*(1+pre[i]*closeprice_change.iloc[i+j]))
            res.append(temp)
        res = np.array(res)
        
        plt.figure(figsize=(10, 6))
        for i in range(res.shape[0]):
            plt.plot(np.arange(res.shape[1]),res[i] - 1,alpha=0.5)
        plt.title(name)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.show()
        return np.mean(res,axis=0)

    def backtestor_sharpeRatio(self, rf=0.03):
        scalerN = 253 * 6.5 * 6 # trading times one yr
        start = self.start
        end = self.start + self.pred_num
        pricechanges = self.X['close'].pct_change().shift(-1).iloc[start: end]
        portfolioReturn = pricechanges * self.pred_array
        sharpeRatio = (portfolioReturn.mean() * scalerN - rf) / (np.sqrt(scalerN) * portfolioReturn.std())
        return sharpeRatio

    # def backtestor_sharpeRatio(self, rf=0.03):
    #     start = self.start
    #     end = self.start + self.pred_num
    #     closeprice_change = self.X['close'].pct_change().shift(-1).iloc[start: end]
    #     self.pnl = (closeprice_change * self.pred_array + 1).cumprod()
    #     sharpeRatio = np.mean(self.pnl - 1 - rf) / np.std(self.pnl)

    #     return sharpeRatio
