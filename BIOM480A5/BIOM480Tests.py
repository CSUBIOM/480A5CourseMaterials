# This file is part of the BIOM480A5 course discussion. 
# Each student will present on a different topic related to statistical tests.
# The goal is to create a function for each topic that can be used to perform the test.
# Each student will create a function with the following format:
# def function_name(parameters):
#     '''
#     Description of the function and its parameters.
#     Parameters:
#     parameter1 : type
#         Description of parameter1.
#     parameter2 : type
#         Description of parameter2.
#     Returns:
#     return_value : type
#         Description of return_value.
#     Example:
#     >>> function_name(parameter1, parameter2)
#     >>> print(return_value)
#     Notes:
#     Additional notes about the function.
#     '''
# The function should be able to be called from the command line or from another script.

# Please enter your tests below in the correct location (look for your name). 
# If you need to add a module, please do so at the top.

import numpy as np
from scipy import stats

# Example will present on the topic of t-test, creating function named 'ttest'
def ttest(a, b):
    '''
    Perform a t-test on two independent samples.
    Parameters:
    a : array_like
        First sample.
    b : array_like
        Second sample.
    Returns:
    t_stat : float
        The calculated t-statistic.
    p_value : float
        The two-tailed p-value.

    Example:
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [2, 3, 4, 5, 6]
    >>> t_stat, p_value = ttest(a, b)
    >>> print(t_stat)
    0.0
    >>> print(p_value)
    0.3465935070873343
    
    Notes:
    This function assumes that the two samples have equal variances.
    The p-value is calculated using the two-tailed test.
    '''
    # Calculate the means and standard deviations
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a, ddof=1)
    std_b = np.std(b, ddof=1)
    n_a = len(a)
    n_b = len(b)
    # Calculate the t-statistic
    t_stat = (mean_a - mean_b) / np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
    # Calculate the degrees of freedom
    df = n_a + n_b - 2
    # Calculate the p-value
    p_value = 2 * (1 - np.abs(t_stat) / np.sqrt(df))
    return t_stat, p_value

#*********************


# Becca B will present on the topic of Breusch-Pagan test, creating function named 'breusch_pagan' 

#*********************
# Ella B will present on the topic of Variance Inflation Factor (VIF) test, creating function named 'vif' 

#*********************
# Lauren B will present on the topic of Jarque-Bera test, creating function named 'jarque_bera' 

#*********************
# James B will present on the topic of Hotelling’s T-squared test, creating function named 'hotelling_t' 

#*********************
# Chayanee C will present on the topic of D’Agostino and Pearson’s test, creating function named 'dagostino' 

#*********************
# Deven D will present on the topic of Chi-square goodness-of-fit test, creating function named 'chi2gof' 

#*********************
# Jackson E will present on the topic of McNemar’s test, creating function named 'mcnemar' 

#*********************
# Delaney E will present on the topic of Ramsey RESET test, creating function named 'reset_test' 

#*********************
# Christian F will present on the topic of Shapiro-Wilk test, creating function named 'shapiro' 

#*********************
# Lillian G will present on the topic of Mantel test, creating function named 'mantel' 

#*********************
# Andy G will present on the topic of Permutation test, creating function named 'permutation_test' 
def permutation_test(a, b, n_permutations=10000, statistic=None):
    '''
    Perform a permutation test to determine if two samples come from the same distribution.
    
    Parameters:
    a : array_like
        First sample.
    b : array_like
        Second sample.
    n_permutations : int, optional
        Number of permutations to perform. Default is 10000.
    statistic : function, optional
        Function that returns the test statistic. Default is the difference in means.
        
    Returns:
    p_value : float
        The estimated p-value for the null hypothesis that the two samples come from the same distribution.
    observed_stat : float
        The observed test statistic.
        
    Example:
    >>> a = [10, 11, 12, 13, 14]
    >>> b = [7, 8, 9, 10, 11]
    >>> p_value, observed_stat = permutation_test(a, b)
    >>> print(f"P-value: {p_value}")
    P-value: 0.0384
    >>> print(f"Observed statistic: {observed_stat}")
    Observed statistic: 3.0
    
    Notes:
    The permutation test is a non-parametric statistical test that estimates the probability of obtaining
    the observed test statistic (or a more extreme value) under the null hypothesis that the two samples
    come from the same distribution. The test works by randomly permuting the combined data many times
    and calculating the test statistic for each permutation.
    
    The test makes no assumptions about the underlying distributions of the data, making it useful
    when the assumptions of parametric tests (like the t-test) are not met.
    '''
    import numpy as np
    
    # Convert inputs to numpy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Default statistic is the difference in means
    if statistic is None:
        statistic = lambda x, y: np.mean(x) - np.mean(y)
    
    # Calculate the observed test statistic
    observed_stat = statistic(a, b)
    
    # Combine the data
    combined = np.concatenate((a, b))
    n_a = len(a)
    n_b = len(b)
    n = n_a + n_b
    
    # Count the number of permutations where the statistic is as extreme as observed
    count = 0
    for _ in range(n_permutations):
        # Randomly permute the combined data
        np.random.shuffle(combined)
        
        # Split the permuted data into two groups of the original sizes
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        
        # Calculate the permuted test statistic
        perm_stat = statistic(perm_a, perm_b)
        
        # Count if the permuted statistic is as extreme as the observed
        if np.abs(perm_stat) >= np.abs(observed_stat):
            count += 1
    
    # Calculate the p-value
    p_value = count / n_permutations
    
    return p_value, observed_stat
#*********************
# Emma G will present on the topic of Anderson-Darling test, creating function named 'anderson' 

#*********************
# Nick G will present on the topic of Boschloo’s test, creating function named 'boschloo' 

#*********************
# Joshua H will present on the topic of Cochran’s Q test, creating function named 'cochran_q' 

#*********************
# Paycen H will present on the topic of Phi coefficient test, creating function named 'phicoeff' 

#*********************
# Kyle H will present on the topic of Fisher’s exact test, creating function named 'fisher_exact' 

#*********************
# Emma H will present on the topic of Spearman correlation test, creating function named 'spearmanr' 

#*********************
# Elijah J will present on the topic of Bartlett’s test, creating function named 'bartlett' 

#*********************
# Gabriela J will present on the topic of Chi-square test of independence, creating function named 'chi2indep' 

#*********************
# Addison L will present on the topic of Permutation test for correlation, creating function named 'permutation_corr' 

#*********************
# Matthew L will present on the topic of Lilliefors test, creating function named 'lilliefors' 

#*********************
# Hassan M will present on the topic of Partial correlation test, creating function named 'partial_corr' 

#*********************
# Bella P will present on the topic of Durbin-Watson test, creating function named 'durbin_watson' 

#*********************
# Chris RT will present on the topic of Fligner-Killeen test, creating function named 'fligner_killeen' 

#*********************
# Mariana S will present on the topic of Point-Biserial correlation test, creating function named 'pointbiserialr' 

#*********************
# Dylan S will present on the topic of Barnard’s exact test, creating function named 'barnard_exact' 

#*********************
# Jacob S will present on the topic of Pearson correlation test, crating function named 'pearsonr' 

#*********************
# Hayley S will present on the topic of Brown-Forsythe test, creating function named 'brown_forsythe' 

#*********************
# Kayla T will present on the topic of Distance correlation test, creating function named 'dcor' 

#*********************
# Vivia VDM will present on the topic of White test, creating function named 'white_test' 

#*********************
# Fig V will present on the topic of Linear regression coefficient test, creating function named 'linreg' 

#*********************
# AbbyMae W will present on the topic of Kendall’s Tau correlation test, creating function named 'kendalltau' 

#*********************
# Alvina Y will present on the topic of F-test for overall regression, creating function named 'f_test' 

#*********************
# Polina Z will present on the topic of Harvey-Collier test, creating function named 'harvey_collier'

#*********************

