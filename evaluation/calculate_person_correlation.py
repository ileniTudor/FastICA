from scipy.stats.stats import pearsonr


def calculate_pearson_correlation(prediction, actual,verbose:bool = False):
    '''
    Caluculate the mean of the pearson correlation coefficient between each predicted sources to the corresponding
    initial sources. The order of the predicted sources is not known, so we assume that the ones that match better
    (having a larger coefficient) are pair signals.

    :param prediction: matrix having 2 lines and (signal size) columns: represent the predicted sources
    :param actual: matrix having 2 lines and (signal size) columns: represent the real (initial) sources
    :param verbose: if True will print each correlation coefficient
    :return: mean of correlation on each sources
    '''
    # Compute the correlation beteen true source 0 and each prediction.
    pear_0_0 = pearsonr(actual[0], prediction[0])
    pear_0_1 = pearsonr(actual[0], prediction[1])
    # The one which is closer (pers_coef is larger) is that one corresponding to the source 0

    if abs(pear_0_0[0]) > abs(pear_0_1[0]):
        # True source 0 match with the prediction 0.
        pear0 = pear_0_0
        # Compute the pear coef for the true source 1 with the prediction 1.
        pear1 = pearsonr(actual[1], prediction[1])
    else:
        # True source 1 match with the prediction 1.
        pear0 = pear_0_1
        # Compute the pear coef for the true source 1 with the prediction 0.
        pear1 = pearsonr(actual[1], prediction[0])
    if verbose:
        print("pear 0:", pear0)
        print("pear 1:", pear1)
    mean_pearson_coeff = (abs(pear0[0]) + abs(pear1[0])) / 2.0
    if verbose:
        print("mean_pearson_coeff",mean_pearson_coeff)
    return mean_pearson_coeff