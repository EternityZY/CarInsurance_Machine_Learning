import json
import os

from imblearn.over_sampling import ADASYN, RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.utils import resample

from DecisionTree.MyDecisionTree import MyDecisionTree
from LogisticRegression.MyLogisticRegression import MyLogisticRegression
from NaiveBayes.MyNaiveBayes import MyGaussianNB

pd.set_option('display.max_rows', 1000)  # 具体的行数或列数可自行设置
pd.set_option('display.max_columns', 1000)


class MyModel():
    def __init__(self):

        self.train_df = pd.read_csv("VI_train.csv")
        self.test_df = pd.read_csv("VI_test.csv")

        self.train_df = self.train_df.drop(['Unnamed: 0'], axis=1)


    # 对训练集和测试集进行标准化
    def standardData(self, X_train, X_valid, X_test):
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        X_valid = sc_X.transform(X_valid)
        return X_train, X_valid, X_test

    def upsampleFeature(self, df):

        def upsample(df, u_feature, n_upsampling):

            df_temp = df.copy()
            ones = df.copy()
            # 根据不同特征，在其周边随机生成
            for n in range(n_upsampling):
                if u_feature == 'Annual_Premium':
                    df_temp[u_feature] = ones[u_feature].apply(
                        lambda x: x + random.randint(-1, 1) * x * 0.05)  # change Annual_premiun in the range of 5%
                else:
                    df_temp[u_feature] = ones[u_feature].apply(
                        lambda x: x + random.randint(-5, 5))  # change Age in the range of 5 years

                if n == 0:
                    df_new = df_temp.copy()
                else:
                    df_new = pd.concat([df_new, df_temp])
            return df_new

        df_train_up_age = upsample(df.loc[df['Response'] == 1], 'Age', 1)
        df_train_up_vintage = upsample(df.loc[df['Response'] == 1], 'Vintage', 1)

        df_ext = pd.concat([df, df_train_up_age])
        df_ext = pd.concat([df_ext, df_train_up_vintage])
        # X_train = df_ext.drop(columns=['Response'])
        # y_train = df_ext.Response
        print(len(df_ext))
        return df_ext

    def upsampleData(self, df):

        ros = RandomOverSampler(random_state=42, sampling_strategy='minority')
        x_train_sampled, y_train_sampled = ros.fit_resample(df.drop('Response', axis=1), df['Response'])

        ada = ADASYN(random_state=42)
        x_train_sampled, y_train_sampled = ada.fit_resample(df.drop('Response', axis=1), df['Response'])

        x_train_sampled['Response'] = y_train_sampled
        print(len(x_train_sampled))
        return x_train_sampled

    def downsample(self, df):
        df_no_response = df[df['Response'] == 0]
        df_response = df[df['Response'] == 1]
        df_no_response_downsampled = resample(df_no_response,
                                              replace=False,
                                              n_samples=int(len(df_response)*2),
                                              random_state=42)
        df_downsample = pd.concat([df_no_response_downsampled, df_response])
        print(len(df_downsample))
        return df_downsample

    def featureEngineer(self, df_train, df_test):
        # 获得特征名
        df_train_response = df_train.loc[df_train.Response == 1].copy()

        categorical_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age',
                                'Vehicle_Damage', 'Policy_Sales_Channel']
        text_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

        # 对于文本特征进行编码
        labelEncoder = preprocessing.LabelEncoder()
        for f in text_features:
            df_train[f] = labelEncoder.fit_transform(df_train[f])
            df_test[f] = labelEncoder.fit_transform(df_test[f])

        # 更改数据类型
        df_train.Region_Code = df_train.Region_Code.astype('int32')
        df_train.Policy_Sales_Channel = df_train.Policy_Sales_Channel.astype('int32')

        df_test.Region_Code = df_test.Region_Code.astype('int32')
        df_test.Policy_Sales_Channel = df_test.Policy_Sales_Channel.astype('int32')

        # 对年龄按照年龄段进行编码
        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        df_train['bin_age'] = pd.cut(df_train['Age'], bins)         # 按照分段分类
        df_train['age_bin_cat'] = labelEncoder.fit_transform(df_train['bin_age'])

        df_test['bin_age'] = pd.cut(df_test['Age'], bins)
        df_test['age_bin_cat'] = labelEncoder.fit_transform(df_test['bin_age'])

        # 删去中间过度特征
        try:
            df_train.drop(columns=['bin_age'], inplace=True)
            df_test.drop(columns=['bin_age'], inplace=True)
        except:
            print('already deleted')

        # 建立新的特征列 按照车龄和是否损坏  因为这两个特征对于是否选择汽车保险十分重要
        df_train['old_damaged'] = df_train.apply(lambda x: pow(2, x.Vehicle_Age) + pow(2, x.Vehicle_Damage), axis=1)
        df_test['old_damaged'] = df_test.apply(lambda x: pow(2, x.Vehicle_Age) + pow(2, x.Vehicle_Damage), axis=1)
        return df_train, df_test

    def preprocessing(self, augmentation='upsampleWithoutValid', normal=True):
        # 根据特征工程 对特征进行预处理
        df_train = self.train_df.copy().set_index('id')
        df_test = self.test_df.copy().set_index('id')

        df_train, df_test = self.featureEngineer(df_train, df_test)
        # 按照训练集80%的比例划分训练集和验证集
        df_temp, X_valid_y, _, y_valid = train_test_split(df_train, df_train['Response'], train_size=0.8,
                                                        random_state=123)

        if augmentation == 'upsampleWithoutValid':
            # 仅上采样正类别
            # 这里不划分验证集，使用所有样本进行训练
            df_train = self.upsampleFeature(df_train)

        elif augmentation=='upsampleWithValid':
            # 划分验证集
            # 注意  验证集中不实用数据增强技术
            df_train = self.upsampleFeature(df_temp)

        elif augmentation=='upsampleData':
            # 使用随机上采样方法
            df_train = self.upsampleData(df_train)

        elif augmentation=='downsample':
            # 根据正样本数量，下采样负样本
            df_train = self.downsample(df_train)

        else:
            # 不执行数据增强
            df_train = df_train

        # 训练集标签
        # df_train = shuffle(df_train)
        # X_valid_y = shuffle(X_valid_y)

        X_train = df_train.drop(columns=['Response'])
        y_train = df_train.Response
        # 获得验证集标签
        X_valid = X_valid_y.drop(columns=['Response'])
        y_valid = X_valid_y.Response
        # 测试集
        X_test= df_test.values

        print(X_train.columns)

        if normal==True:
            # 标准化数据
            X_train, X_valid, X_test = self.standardData(X_train, X_valid, X_test)

        print('Train set target class count with over-sampling:')
        print(y_train.value_counts())
        print('Test set target class count with over-sampling:')
        print(len(X_test))

        return X_train, np.array(y_train), X_valid, np.array(y_valid), X_test


    def plot_ROC(self, fpr, tpr, m_name):
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.figure(figsize=(15, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('Receiver operating characteristic for %s' % m_name, fontsize=20)
        plt.legend(loc="lower right", fontsize=16)
        # plt.show()
        plt.savefig('DecisionTree/{:s}.png'.format(m_name))

    def writeResults(self, path, predictions):
        res = dict()
        for i in range(len(predictions)):
            res[str(i)] = int(predictions[i])
        with open(os.path.join(path, 'submission.json'), 'w', encoding='UTF-8') as fp:
            fp.write(json.dumps(res, indent=2, ensure_ascii=False))
        print("成功写入文件。")

    def MXGBoost(self, valid=True, normal=True, search=True):
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='downsample',
                                                                                normal=normal)
        if search==False:

            XGB_model = XGBClassifier(random_state=1970, max_depth=7, reg_lambda=1.2, reg_alpha=1.2,
                                      min_child_weight=1, n_estimators=130,
                                      objective='binary:logistic',
                                      learning_rate=0.15, gamma=0.3, colsample_bytree=0.5,
                                      eval_metric='auc')

            # 模型训练
            XGB_model.fit(X_train, y_train)

            if valid==True:
                # 概率预测
                XGB_preds = XGB_model.predict_proba(X_valid)
                # 绘制ROC_AUC曲线
                XGB_score = roc_auc_score(y_valid, XGB_preds[:, 1])
                (fpr, tpr, thresholds) = roc_curve(y_valid, XGB_preds[:, 1])
                self.plot_ROC(fpr, tpr, 'XGBoost')
                print('ROC AUC score for XGBoost model with over-sampling + 2 new features: %.4f' % XGB_score)

                # 计算F1 Score
                XGB_class = XGB_model.predict(X_valid)
                print('XGBoost F1 score: %0.4f' % f1_score(y_valid, XGB_class))
                print('XGBoost Reports\n', classification_report(y_valid, XGB_class))
                # 查看验证集的混淆矩阵
                confusion_matrix(y_valid, XGB_class)
                # 查看重要特征
                xgb.plot_importance(XGB_model, importance_type='gain')
                plt.show()

            # 不做验证，直接测试集并保存结果
            # 计算F1 score
            XGB_class = XGB_model.predict(X_test)
            xgb.plot_importance(XGB_model)
            plt.show()

            path = 'XGBoost'
            self.writeResults(path, XGB_class)
        else:

            param_dist = {
                'n_estimators': range(80, 200, 20),
                'max_depth': range(5, 15, 1),
                'learning_rate': np.linspace(0.01, 0.1, 10),
                'subsample': np.linspace(0.7, 0.9, 10),
                'colsample_bytree': np.linspace(0.5, 0.98, 10),
                'min_child_weight': range(1, 9, 1)
            }

            other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 10, 'min_child_weight': 1, 'seed': 0,
                            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
            model = xgb.XGBClassifier(**other_params)
            optimized_GBM = GridSearchCV(estimator=model, param_grid=param_dist, scoring='f1', cv=5, verbose=1,
                                         n_jobs=4)
            optimized_GBM.fit(X_train, y_train)

            print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
            print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))



    def MRandomForest(self, valid=True, normal=True):
        # 随机森林
        rf_params = {'max_depth': 20, 'n_estimators': 150, 'min_samples_leaf': 1}
        rf_params['random_state'] = 123
        rf = RandomForestClassifier(**rf_params)
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='downsample',
                                                                                normal=normal)
        # 模型训练
        rf.fit(X_train, y_train)

        if valid==True:
            rf_preds = rf.predict(X_valid)
            rf_preds_prob = rf.predict_proba(X_valid)[:, 1]
            reg_score_uc = roc_auc_score(y_valid, rf_preds_prob)
            print('ROC AUC score for Random Forest model with over-sampling: %.4f' % reg_score_uc)
            print('Random Forest f1-score', f1_score(y_valid, rf_preds))
            print('Random Forest Reports\n', classification_report(y_valid, rf_preds))


        # 不做验证，直接测试集并保存结果
        rf_preds = rf.predict(X_test)

        # 显示重要特征
        title = "Feature Importances Random Forest"
        feat_imp = pd.DataFrame({'importance': rf.feature_importances_})
        feat_imp['feature'] = X_train.columns
        feat_imp.sort_values(by='importance', ascending=True, inplace=True)
        feat_imp = feat_imp.set_index('feature', drop=True)
        feat_imp.plot.barh(title=title, figsize=(8, 8))
        plt.xlabel('Feature Importance Score')
        plt.show()

        path = 'RandomForest'
        self.writeResults(path, rf_preds)


    def MLogisticRegression(self, valid=True, normal=True, sklearn=False):
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='downsample',
                                                                                normal=normal)

        if sklearn==True:
            lr = LogisticRegression()

            lr.fit(X_train, y_train)

            if valid==True:
                lr_preds = lr.predict(X_valid)
                lf_preds_prob = lr.predict_proba(X_valid)[:, 1]

                reg_score_uc = roc_auc_score(y_valid, lf_preds_prob)
                print('ROC AUC score for LogisticRegression model with over-sampling: %.4f' % reg_score_uc)
                print('Logistic Regression f1-score', f1_score(y_valid, lr_preds))
                print('Logistic regression Reports\n', classification_report(y_valid, lr_preds))

            lr_preds = lr.predict(X_test)

            path = 'LogisticRegression'
            self.writeResults(path, lr_preds)
        else:
            lr = MyLogisticRegression()
            lr.fit(X_train, y_train, X_valid, y_valid, method='batch',learning_rate=0.05, epochs=501, tol=1e-7)

            if valid == True:
                lr_preds = lr.predict(X_valid)
                lf_preds_prob = lr.predict_proba(X_valid)

                reg_score_uc = roc_auc_score(y_valid, lf_preds_prob)
                print('ROC AUC score for LogisticRegression model with over-sampling: %.4f' % reg_score_uc)
                print('Logistic Regression f1-score', f1_score(y_valid, lr_preds))
                print('Logistic regression Reports\n', classification_report(y_valid, lr_preds))

            lr_preds = lr.predict(X_test)

            path = 'LogisticRegression'
            self.writeResults(path, lr_preds)


    def MDecisionTree(self, valid=True, normal=True, sklearn=False):
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='downsample',
                                                                                normal=normal)



        if sklearn==True:

            DT = DecisionTreeClassifier(criterion='gini')  # {"gini", "entropy"}
            DT.fit(X_train, y_train)

            if valid == True:
                dt_preds = DT.predict(X_valid)
                dt_preds_prob = DT.predict_proba(X_valid)[:, 1]

                reg_score_uc = roc_auc_score(y_valid, dt_preds_prob)
                print('ROC AUC score for Decision Tree model with over-sampling: %.4f' % reg_score_uc)
                print('Decision Tree f1-score', f1_score(y_valid, dt_preds))
                print('Decision Tree Reports\n', classification_report(y_valid, dt_preds))

            dt_preds = DT.predict(X_test)

            path = 'DecisionTree'
            self.writeResults(path, dt_preds)

        if sklearn==False:
            f1_score_list = []
            res_json = []

            max_f1=0
            good_deep = 0
            max_res=[]
            start = 8
            end = 10
            X_train = np.array(X_train)
            X_valid = np.array(X_valid)
            X_test = np.array(X_test)
            for deep in range(start, end):
                # max_depth=4, min_samples_split=2
                DT = MyDecisionTree()
                DT.fit(X_train, y_train, max_depth=deep, min_samples_split=5)
                dt_preds = DT.predict(X_valid)
                dt_preds_prob = DT.predict_proba(X_valid)

                f1 = f1_score(y_valid, dt_preds)
                if f1>max_f1:
                    max_f1=f1
                    max_res=dt_preds
                    good_deep=deep

                res_json.append(dt_preds)
                f1_score_list.append(f1)

                print('max_depth={:d}'.format(deep))
                print('Decision Tree f1-score', f1)
                print('Decision Tree Reports\n', classification_report(y_valid, dt_preds))

                (fpr, tpr, thresholds) = roc_curve(y_valid, dt_preds_prob)
                self.plot_ROC(fpr, tpr, 'DT max_depth={:d}, F1={:.4f}'.format(deep, f1))

                del DT
                print('\n\n\n')

            plt.plot(range(start, end), f1_score_list, c='r')
            plt.xlabel('Max Depth')
            plt.ylabel('F1 Score')
            plt.title('F1 Score with Max Depth')
            plt.savefig('DecisionTree/resPlot.png')


            DT = MyDecisionTree()
            DT.fit(X_train, y_train, max_depth=good_deep, min_samples_split=5)
            dt_preds = DT.predict(X_test)
            path = 'DecisionTree'
            self.writeResults(path, dt_preds)





    def MSVM(self, valid=True, normal=True):
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='upsampleWithoutValid',
                                                                                normal=normal)

        svm = SVC(random_state = 42,kernel = 'rbf', probability=True)
        svm.fit(X_train, y_train)

        if valid == True:
            svm_preds = svm.predict(X_valid)
            svm_preds_prob = svm.predict_proba(X_valid)[:, 1]

            reg_score_uc = roc_auc_score(y_valid, svm_preds_prob)
            print('ROC AUC score for SVM model with over-sampling: %.4f' % reg_score_uc)
            print('SVM f1-score', f1_score(y_valid, svm_preds))
            print('SVM Reports\n', classification_report(y_valid, svm_preds))

        svm_preds = svm.predict(X_test)

        path = 'SVM'
        self.writeResults(path, svm_preds)


    def MGaussianNB(self, valid=True, normal=True, sklearn=False):
        X_train, y_train, X_valid, y_valid, X_test = self.preprocessing(augmentation='downsample',
                                                                                normal=normal)

        if sklearn==True:
            NaiveBayes = GaussianNB()
            NaiveBayes.fit(X_train, y_train)

            if valid == True:
                NB_preds = NaiveBayes.predict(X_valid)
                NB_preds_prob = NaiveBayes.predict_proba(X_valid)[:, 1]

                reg_score_uc = roc_auc_score(y_valid, NB_preds_prob)
                print('ROC AUC score for NaiveBayes model with over-sampling: %.4f' % reg_score_uc)
                print('NaiveBayes f1-score', f1_score(y_valid, NB_preds))
                print('NaiveBayes Reports\n', classification_report(y_valid, NB_preds))

            NB_preds = NaiveBayes.predict(X_test)

        else:
            X_train = np.array(X_train)
            X_valid = np.array(X_valid)
            X_test = np.array(X_test)
            NaiveBayes = MyGaussianNB()
            NaiveBayes.fit(X_train, y_train)

            if valid == True:
                NB_preds = NaiveBayes.predict(X_valid)
                print('Naive Bayes f1-score', f1_score(y_valid, NB_preds))
                print('Naive Bayes Reports\n', classification_report(y_valid, NB_preds))

            NB_preds = NaiveBayes.predict(X_test)


        path = 'NaiveBayes'
        self.writeResults(path, NB_preds)


if __name__ == '__main__':
    model = MyModel()
    model.MXGBoost(valid=True, normal=False, search=False)
    model.MRandomForest(valid=True, normal=False)
    model.MLogisticRegression(valid=True, sklearn=True)
    model.MDecisionTree(valid=True, normal=True, sklearn=False)
    #
    model.MGaussianNB(valid=True, normal=True, sklearn=False)
    model.MSVM(valid=True)


