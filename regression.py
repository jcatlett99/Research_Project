import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


class Regression:

    def __init__(self):
        pass

    @staticmethod
    def logistic_regression(path):

        data = pd.read_csv(path, index_col=0)
        model = smf.logit(formula='aesthetic ~ novelty + '
                                  'histograms + entropy + '
                                  'straight_diagonal_line_ratio + '
                                  'horizontal_vertical_line_ration + '
                                  'diagonal_dominance + '
                                  'symmetry + rule_of_thirds_power_points + '
                                  'rule_of_thirds_gridlines + sharpness + '
                                  'contrast + luminance + saturation', data=data).fit()
        print(model.summary())

    @staticmethod
    def linear_regression(features, path):
        data = pd.read_csv(path, index_col=0)
        X = data[features]
        Y = data[['aesthetic']]

        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        # predictions = model.predict(X)

        print(model.summary())
