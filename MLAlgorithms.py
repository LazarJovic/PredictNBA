import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seabornInstance
import shap
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


class MLAlgorithms:
    def __init__(self):
        pass

    def is_not_used(self):
        pass

    def drop_features(self, df):
        self.is_not_used()
        fields_to_drop = ["gmDate", "homeTeam", "teamConf", "teamDiv", "teamLoc", "teamRslt",
                          "awayTeam", "opptConf", "opptDiv", "opptLoc", "opptRslt", "ptsDifference", "homeWin"]
        X = df.drop(fields_to_drop, axis=1).dropna()
        return X

    # SVM
    def svm_func(self, df, test_df):
        self.is_not_used()

        X_train = self.drop_features(df)
        y_train = df["homeWin"].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)

        X_test = self.drop_features(test_df)
        y_test = test_df["homeWin"].values
        X_test = scaler.fit_transform(X_test)

        clf = svm.SVC(kernel="linear")

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        print("metrics.accuracy SVM: " + str(metrics.accuracy_score(y_test, pred)))
        print("Confusion matrix for SVM: \n")
        print(confusion_matrix(y_test, pred))
        print("Classification report for SVM: \n")
        print(classification_report(y_test, pred))
        print("\n\n")

    # Linear regression
    def linear_regression_func(self, df, test_df):
        self.is_not_used()

        X_train = self.drop_features(df)
        y_train = df["ptsDifference"].values

        # Razlike u poenima
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        seabornInstance.distplot(df["ptsDifference"])
        plt.title("Razlika u poenima")
        plt.show()

        X_test = self.drop_features(test_df)
        y_test = test_df["ptsDifference"]

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Optimalni koeficijenti za sve atribute
        print("Linear Regression parameters: \n")
        coeff_df = pd.DataFrame(regressor.coef_, X_test.columns, columns=['Coefficient'])
        print(coeff_df)

        y_pred = regressor.predict(X_test)
        df1 = pd.DataFrame({"Stvarni rezultat": y_test, "Prediktovani rezultat": y_pred})
        print("Prikaz stvarne i prediktovane razlike u poenima: \n")
        print(df1)

        scores2 = regressor.score(X_test, y_test)

        print("R2 score LR: " + str(scores2))
        print("\n\n")

    # RandomForestClassifier1
    def random_forest_func(self, df, test_df):
        self.is_not_used()

        X_train = self.drop_features(df)
        y_train = df["homeWin"].values

        X_test = self.drop_features(test_df)
        y_test = test_df["homeWin"]

        rf = RandomForestClassifier(random_state=0, n_estimators=100, bootstrap=True, max_features="sqrt")
        rf.fit(X_train, y_train)

        shap.initjs()
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)

        y_pred_rf = rf.predict(X_test)
        print("metrics.accuracy RF: " + str(metrics.accuracy_score(y_test, y_pred_rf)))
        print("Confusion matrix for RF: \n")
        print(confusion_matrix(y_test, y_pred_rf))
        print("Classification report for RF: \n")
        print(classification_report(y_test, y_pred_rf))
        print("\n\n")

    # Naive Bayes
    def naive_bayes(self, df, test_df):
        self.is_not_used()

        X_train = self.drop_features(df)
        y_train = df["homeWin"].values

        X_test = self.drop_features(test_df)
        y_test = test_df["homeWin"]

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred_nb = gnb.predict(X_test)
        print("metrics.accuracy NB: " + str(metrics.accuracy_score(y_test, y_pred_nb)))
        print("Confusion matrix for RF: \n")
        print(confusion_matrix(y_test, y_pred_nb))
        print("Classification report for RF: \n")
        print(classification_report(y_test, y_pred_nb))
        print("\n\n")
