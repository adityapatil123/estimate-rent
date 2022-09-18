import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score


def read_csv(csv_path):
    return pd.read_csv(csv_path)


def get_dummy_data(df):
    return pd.get_dummies(df)


def train_test_and_split(df_dum: pd.DataFrame):
    x = df_dum.drop('Rent', axis=1)
    y = df_dum.Rent.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def multi_linear_regression(x_train, y_train):
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    return np.mean(cross_val_score(lm, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))


def lasso_regression(x_train, y_train):
    lm_l = Lasso(alpha=.13)
    lm_l.fit(x_train, y_train)
    return np.mean(cross_val_score(lm_l, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))


def get_alpha_and_error(x_train, y_train):
    alpha = []
    error = []

    for i in range(1, 100):
        alpha.append(i / 100)
        lml = Lasso(alpha=(i / 100))
        error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

    # plt.plot(alpha, error)

    err = tuple(zip(alpha, error))
    df_err = pd.DataFrame(err, columns=['alpha', 'error'])
    return df_err[df_err.error == max(df_err.error)]


def random_forest_regression(x_train, y_train):
    rf = RandomForestRegressor()
    return np.mean(cross_val_score(rf, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))


def tune_models_grid_search_cv(rf, x_train, y_train):
    parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('mse', 'mae'),
                  'max_features': ('auto', 'sqrt', 'log2')}

    gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
    gs.fit(x_train, y_train)
    return gs


def test_ensembles(lm, lm_l, gs, x_test, y_test):
    tpred_lm = lm.predict(x_test)
    tpred_lml = lm_l.predict(x_test)
    tpred_rf = gs.best_estimator_.predict(x_test)

    from sklearn.metrics import mean_absolute_error
    mean_absolute_error(y_test, tpred_lm)
    mean_absolute_error(y_test, tpred_lml)
    mean_absolute_error(y_test, tpred_rf)
    mean_absolute_error(y_test, (tpred_lm + tpred_rf) / 2)


def predict_from_gs_model(gs, x_test):
    pickl = {'model': gs.best_estimator_}
    pickle.dump(pickl, open('model_file' + ".p", "wb"))

    file_name = "model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']

    return model.predict(np.array(list(x_test.iloc[1, :])).reshape(1, -1))[0]


def model_data():
    df = read_csv('dataset/Cleaned_House_Rent_Dataset.csv')
    df_dum = get_dummy_data(df)
    x_train, x_test, y_train, y_test = train_test_and_split(df_dum)
    lm = multi_linear_regression(x_train, y_train)
    lm_l = lasso_regression(x_train, y_train)
    rf = random_forest_regression(x_train, y_train)
    gs = tune_models_grid_search_cv(rf, x_train, y_train)
    res = predict_from_gs_model(gs, x_test)
    print(res)


if __name__ == "__main__":
    model_data()