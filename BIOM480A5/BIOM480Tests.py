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

def vif(X, return_dataframe=True):
   """
   Calculate Variance Inflation Factor (VIF) for each variable in a dataset using scikit-learn.
  
   Parameters:
   -----------
   X : pandas.DataFrame
       The input data containing predictor variables
   return_dataframe : bool, default=True
       If True, returns a pandas DataFrame with VIF values for each variable.
       If False, returns a dictionary.
  
   Returns:
   --------
   pandas.DataFrame or dict
       VIF values for each variable. Higher values (>5 or >10) indicate multicollinearity.
   """
   import pandas as pd
   from sklearn.linear_model import LinearRegression
  
   # Converts to DataFrame if not already
   if not isinstance(X, pd.DataFrame):
       X = pd.DataFrame(X)
  
   # Calculate VIF for each variable
   vif_data = {}
   for col in X.columns:
       # Features that will be used as predictors (all except the current one)
       features = [x for x in X.columns if x != col]
      
       # Fit linear regression
       X_features = X[features]
       y = X[col]
       reg = LinearRegression().fit(X_features, y)
      
       # Calculate R-squared
       r2 = reg.score(X_features, y)
      
       # Calculate VIF
       vif_data[col] = 1 / (1 - r2)
  
   if return_dataframe:
       vif_df = pd.DataFrame({
           "Variable": list(vif_data.keys()),
           "VIF": list(vif_data.values())
       })
       return vif_df
  
   return vif_data

#*********************
# Lauren B will present on the topic of Jarque-Bera test, creating function named 'jarque_bera' 

#*********************
# James B will present on the topic of Hotelling’s T-squared test, creating function named 'hotelling_t' 

#*********************
# Chayanee C will present on the topic of D’Agostino and Pearson’s test, creating function named 'dagostino' 

import numpy as np
from scipy.stats import skew, kurtosis, skewtest, kurtosistest, normaltest

def dagostino(data):
    '''
    Perform D'Agostino and Pearson's test for normality and return detailed diagnostics.

    Parameters:
    data : array_like
        A one-dimensional array of sample data.

    Returns:
    results : dict
        Dictionary containing:
        - 'k2_statistic': combined test statistic
        - 'p_value': p-value of the test
        - 'skewness': sample skewness
        - 'kurtosis': sample excess kurtosis
        - 'z_skew': skewness Z-score
        - 'z_kurt': kurtosis Z-score

    Example:
    >>> data = [-1.83, -0.58, 0.70, -1.35, 0.89, -0.18, 2.88, 0.94, -0.43, 0.27]
    >>> result = dagostino(data)
    >>> for key, val in result.items():
            print(f"{key}: {val:.4f}")

    Notes:
    This function evaluates both skewness and kurtosis to test for normality using the 
    D'Agostino and Pearson test. It provides individual shape measures as well 
    as the combined test statistic and p-value.
    '''
    # Calculate skewness and kurtosis
    skew_val = skew(data)
    kurt_val = kurtosis(data)  # excess kurtosis by default

    # Get Z-scores for skewness and kurtosis
    z_skew, _ = skewtest(data)
    z_kurt, _ = kurtosistest(data)

    # Combined K² test
    k2_stat, p_value = normaltest(data)

    # Return all results as a dictionary
    return {
        'K2_statistic': k2_stat,
        'p_value': p_value,
        'Skewness': skew_val,
        'Excess Kurtosis': kurt_val,
        'Z-score (skewness)': z_skew,
        'Z-score (kurtosis)': z_kurt
    }

#*********************
# Deven D will present on the topic of Chi-square goodness-of-fit test, creating function named 'chi2gof' 
def chi2gof(observed, expected=None, p_vals=None, ddof=0):
    '''
    Perform a chi-square goodness-of-fit test.
    
    Parameters:
    observed : array_like
        Observed frequencies in each category.
    expected : array_like, optional
        Expected frequencies in each category. If None, a uniform distribution is assumed.
    p_vals : array_like, optional
        Probability for each category under null hypothesis. If provided, expected is calculated as p_vals * sum(observed).
    ddof : int, optional
        Delta degrees of freedom. Number of parameters estimated to calculate the expected frequencies.
        
    Returns:
    chi2_stat : float
        The chi-square test statistic.
    p_value : float
        The p-value for the test.
    
    Example:
    >>> observed = [16, 18, 16, 14, 12, 12]  # Observed frequencies of die rolls
    >>> p_vals = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # Expected probabilities for fair die
    >>> chi2_stat, p_value = chi2gof(observed, p_vals=p_vals)
    >>> print(f"Chi-square statistic: {chi2_stat:.4f}")
    Chi-square statistic: 2.0000
    >>> print(f"p-value: {p_value:.4f}")
    p-value: 0.8491
    
    Notes:
    - For the test to be valid, each expected frequency should typically be at least 5.
    - The null hypothesis is that the observed frequencies match the expected frequencies.
    - A low p-value suggests rejecting the null hypothesis, indicating the data doesn't fit the expected distribution.
    '''
    # Convert to numpy arrays for easier manipulation
    observed = np.asarray(observed)
    
    # Calculate expected frequencies if not provided
    if expected is None:
        if p_vals is not None:
            p_vals = np.asarray(p_vals)
            if not np.isclose(np.sum(p_vals), 1.0):
                raise ValueError("Probabilities must sum to 1")
            expected = p_vals * np.sum(observed)
        else:
            # Uniform distribution if neither expected nor p_vals provided
            expected = np.ones_like(observed) * np.sum(observed) / len(observed)
    else:
        expected = np.asarray(expected)
    
    # Check that observed and expected have the same shape
    if observed.shape != expected.shape:
        raise ValueError("Observed and expected arrays must have the same shape")
    
    # Check that all expected frequencies are > 0
    if np.any(expected <= 0):
        raise ValueError("All expected frequencies must be positive")
    
    # Calculate chi-square statistic
    chi2_stat = np.sum(((observed - expected) ** 2) / expected)
    
    # Calculate degrees of freedom
    df = len(observed) - 1 - ddof
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value
#*********************
# Jackson E will present on the topic of McNemar’s test, creating function named 'mcnemar' 
import scipy.stats as stats

def mcnemar(table):
    """
    Performs McNemar's test on a 2x2 contingency table.
    
    Parameters:
    table (list of lists or 2D array): A 2x2 matrix representing paired binary outcomes.
                                       Format: [[a, b],
                                                [c, d]]
                                       Where:
                                       - a: both tests negative
                                       - b: test1 negative, test2 positive
                                       - c: test1 positive, test2 negative
                                       - d: both tests positive
    
    Returns:
    float: p-value from McNemar's test 

    Example:
    >>> table = [[10, 5],
                [1, 14]]
    >>> p_value = mcnemar(table)
    >>> print(p_value)
    0.22067136191984324

    Notes: 
    McNemar's test is used for paired nominal data. 
    Because it is used with paired data, the sum of a and b must equal the sum of c and d.
    The null hypothesis is that the proportions of discordant pairs are equal.
    If the table is not 2x2, a ValueError will be raised.
    This assumes that the input is a 2x2 matrix, and the DOF is 1.

    """
    # Check if the input is a 2x2 matrix
    if len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2:
        raise ValueError("Input must be a 2x2 matrix.")

# Extract the values from the table
    b = table[0][1]
    c = table[1][0]
    
    # Calculate the test statistic
    stat = (abs(b - c) - 1)**2 / (b + c)
    p_value = stats.chi2.sf(stat, df=1)


    return p_value

#*********************
# Delaney E will present on the topic of Ramsey RESET test, creating function named 'reset_test' 

# The Ramsey RESET test is used to check for model misspecification in regression analysis.
import statsmodels.api as sm
import pandas as pd

def reset_test(model, degree=2):
    '''
    Performs the Ramsey RESET test for model specification.

    Parameters:
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A fitted OLS model from statsmodels.
    degree : int
        The maximum power of the fitted values to include in the test (e.g., 2 includes yhat^2, 3 includes yhat^3, etc.).

    Returns:
    reset_result : dict
        A dictionary containing the F-statistic, p-value, and null hypothesis statement.

    Example:
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris

    >>> # Load the Iris dataset
    >>> data = load_iris()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['species'] = data.target

    >>> # Define independent and dependent variables
    >>> X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    >>> X = sm.add_constant(X)  # add intercept
    >>> y = df['species']  # species as dependent variable

    >>> # Fit the initial linear regression model
    >>> model = sm.OLS(y, X).fit()

    >>> # Call the Ramsey RESET test with the degree argument
    >>> result = reset_test(model, degree=3)

    >>> # Print the test result
    >>> print("Ramsey RESET Test Result:")
    >>> for key, value in result.items():
    >>>     print(f"{key}: {value}")
    
    Ramsey RESET Test Result:
    F-statistic: <value>
    p-value: <value>
    df_diff: <value>
    Null Hypothesis: Model is correctly specified. (No omitted nonlinear terms)

    Notes:
    The Ramsey RESET test detects general misspecification in a linear regression model
    by adding powers of the fitted values and testing their joint significance.
    A low p-value suggests the model may be missing nonlinear components.
    '''
    
    # Check if the model is fitted/get fitted values
    yhat = model.fittedvalues
    # Create new dataframe with powers of fitted values
    df = model.model.exog.copy()
    for d in range(2, degree + 1):
        df = pd.DataFrame(df, columns=model.model.exog_names)  # Convert to DataFrame to append new columns
        # Add new columns for each power of fitted values
        df[f'yhat^{d}'] = yhat**d

    # Fit a new model with the original predictors and the new polynomial terms
    y = model.model.endog
    new_model = sm.OLS(y, sm.add_constant(df)).fit()

    # Perform the F-test for the new model against the original model
    f_test = new_model.compare_f_test(model)

    # Extract the F-statistic and p-value
    reset_result = {
        'F-statistic': f_test[0],
        'p-value': f_test[1],
        'df_diff': int(f_test[2]),
        'Null Hypothesis': 'Model is correctly specified. (No omitted nonlinear terms)'
    }

    return reset_result

#*********************
# Christian F will present on the topic of Shapiro-Wilk test, creating function named 'shapiro' 

#*********************
# Lillian G will present on the topic of Mantel test, creating function named 'mantel' 

def mantel_test(matrix_a, matrix_b, permutations=999, random_state=None):
    '''
    Performs a Mantel test to measure correlation between two distance matrices.
    
    Parameters:
    -----------
    matrix_a : array-like
        First distance matrix (square, symmetric)
    matrix_b : array-like
        Second distance matrix (square, symmetric)
    permutations : int, optional (default=999)
        Number of permutations for the test
    random_state : int or None, optional (default=None)
        Random seed for reproducibility
    
    Returns:
    --------
    r : float
        Mantel correlation statistic
    p_value : float
        p-value from the permutation test
        
    Example:
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist, squareform
    >>> # Create two correlated distance matrices
    >>> coords = np.random.rand(10, 2)
    >>> matrix_a = squareform(pdist(coords, 'euclidean'))
    >>> matrix_b = matrix_a * 0.9 + np.random.rand(10, 10) * 0.1
    >>> matrix_b = (matrix_b + matrix_b.T) / 2  # Make symmetric
    >>> np.fill_diagonal(matrix_b, 0)  # Zero diagonal
    >>> r, p = mantel_test(matrix_a, matrix_b, permutations=99, random_state=42)
    >>> print(f"Mantel statistic r: {r:.4f}, p-value: {p:.4f}")
    Mantel statistic r: 0.9723, p-value: 0.0100
    
    Notes:
    This test evaluates the correlation between two distance matrices using permutation
    to assess statistical significance. It's commonly used in ecology to test for relationships
    between different types of distances (e.g., geographic vs. community composition).
    The matrices must be square, symmetric, and of the same dimensions.
    '''
    # Convert inputs to numpy arrays
    matrix_a = np.asarray(matrix_a)
    matrix_b = np.asarray(matrix_b)
    
    # Check that matrices are square and of the same size
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Distance matrices must be of the same shape")
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Flatten the distance matrices, ignoring the diagonal and keeping only the lower triangle
    # This avoids double-counting distances and including self-distances
    n = matrix_a.shape[0]
    indices = np.triu_indices(n, k=1)
    a_flat = matrix_a[indices]
    b_flat = matrix_b[indices]
    
    # Calculate the Mantel statistic (correlation coefficient)
    r_obs = np.corrcoef(a_flat, b_flat)[0, 1]
    
    # Permutation test
    r_perm = np.zeros(permutations)
    for i in range(permutations):
        # Shuffle one of the matrices
        indices_perm = np.random.permutation(n)
        b_perm = matrix_b[indices_perm, :][:, indices_perm]
        b_flat_perm = b_perm[indices]
        
        # Calculate correlation for permuted matrix
        r_perm[i] = np.corrcoef(a_flat, b_flat_perm)[0, 1]
    
    # Calculate p-value as proportion of permutations with a correlation
    # greater than or equal to the observed correlation
    p_value = np.sum(np.abs(r_perm) >= np.abs(r_obs)) / permutations
    
    return r_obs, p_value
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

def boschloo(table, alternative='two-sided'):
    '''
    Perform Boschloo's exact test on a 2x2 contingency table.
    
    Boschloo's test is a more powerful alternative to Fisher's exact test for 2x2 contingency tables.
    It uses Fisher's p-value as a test statistic and finds the probability of obtaining
    a test statistic at least as extreme as the observed one.
    
    Parameters:
    -----------
    table : array_like
        A 2x2 contingency table. Elements should be non-negative integers.
        The table should be in the format:
            [[a, b], [c, d]]
        where a, b, c, d are the cell counts.
    
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        - 'two-sided': the odds ratio of the underlying population is not one
        - 'less': the odds ratio of the underlying population is less than one
        - 'greater': the odds ratio of the underlying population is greater than one
    
    Returns:
    --------
    statistic : float
        The calculated test statistic - in this case, the p-value from Fisher's exact test.
    
    p_value : float
        The p-value for the Boschloo test, representing the probability of obtaining
        a distribution at least as extreme as the observed one, assuming the null
        hypothesis is true.
    
    Example:
    --------
    >>> table = [[74, 31], [43, 32]]
    >>> statistic, p_value = boschloo(table, alternative="greater")
    >>> print(statistic)
    0.0483  # This is the Fisher's p-value
    >>> print(p_value)
    0.0355  # This is the Boschloo's p-value
    
    Notes:
    ------
    - This function relies on scipy.stats.fisher_exact to calculate the Fisher's exact test p-value
    - Boschloo's test is appropriate when the row sums are fixed but the column sums are not
    - Unlike Fisher's test, Boschloo's test is not symmetric: transposing the table will generally
      give different results
    - Boschloo's test is uniformly more powerful than Fisher's exact test
    '''
    import numpy as np
    from scipy import stats
    
    # Convert input to numpy array
    table = np.asarray(table, dtype=np.int64)
    
    # Check dimensions
    if table.shape != (2, 2):
        raise ValueError("The input table must be a 2x2 contingency table")
    
    # Check for negative values
    if np.any(table < 0):
        raise ValueError("All elements in the table must be non-negative")
    
    # Extract values from the table
    a, b = table[0]
    c, d = table[1]
    
    # Get row and column sums
    row_sums = np.sum(table, axis=1)  # [a+b, c+d]
    col_sums = np.sum(table, axis=0)  # [a+c, b+d]
    n = np.sum(table)  # a+b+c+d
    
    # Calculate the test statistic (Fisher's exact p-value)
    _, fisher_p = stats.fisher_exact(table, alternative=alternative)
    statistic = fisher_p  # The statistic is Fisher's p-value
    
    # Generate all possible tables with the same row sums
    def generate_possible_tables():
        tables = []
        # Maximum possible value for a is min(row_sum[0], col_sum[0])
        max_a = min(row_sums[0], col_sums[0])
        
        for i in range(max_a + 1):
            new_a = i
            new_b = row_sums[0] - new_a
            new_c = col_sums[0] - new_a
            new_d = row_sums[1] - new_c
            
            # Check if the table is valid (all entries should be non-negative)
            if new_b >= 0 and new_c >= 0 and new_d >= 0:
                tables.append(np.array([[new_a, new_b], [new_c, new_d]]))
        
        return tables
    
    # Calculate the p-values for all possible tables
    observed_fisher_p = statistic
    possible_tables = generate_possible_tables()
    
    # Calculate probabilities of each possible table under the null hypothesis
    # We use the maximum likelihood estimate of the common binomial parameter
    # which is (a + c) / (a + b + c + d) under H0: p1 = p2
    p_common = col_sums[0] / n
    
    table_probs = []
    for table in possible_tables:
        a_t, b_t = table[0]
        c_t, d_t = table[1]
        
        # Calculate probability of this table using binomial distributions
        # Pr(table) = Pr(a | n1, p) * Pr(c | n2, p)
        # where n1 = a+b, n2 = c+d, p = (a+c)/(a+b+c+d)
        prob_a = stats.binom.pmf(a_t, row_sums[0], p_common)
        prob_c = stats.binom.pmf(c_t, row_sums[1], p_common)
        table_probs.append(prob_a * prob_c)
    
    # Get Fisher's p-value for each possible table
    fisher_p_values = []
    for possible_table in possible_tables:
        _, p_value = stats.fisher_exact(possible_table, alternative=alternative)
        fisher_p_values.append(p_value)
    
    # Calculate Boschloo's p-value as the sum of probabilities of tables
    # with Fisher's p-value <= the observed Fisher's p-value
    boschloo_p = sum(prob for prob, p_val in zip(table_probs, fisher_p_values) 
                     if p_val <= observed_fisher_p)
    
    return statistic, boschloo_p
#*********************
# Joshua H will present on the topic of Cochran’s Q test, creating function named 'cochran_q' 

#*********************
# Paycen H will present on the topic of Phi coefficient test, creating function named 'phicoeff' 

import numpy as np
from scipy.stats import chi2_contingency

def phicoeff(table):
    """
    Calculate the Phi coefficient from a 2x2 contingency table.

    Parameters:
        table (list of lists or 2x2 np.ndarray): [[a, b], [c, d]]
    
    Returns:
        phi (float): Phi coefficient
        p_value (float): Chi-squared test p-value
    """
    table = np.array(table)
    if table.shape != (2, 2):
        raise ValueError("Input must be a 2x2 contingency table.")
    
    a, b = table[0]
    c, d = table[1]

    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

    phi = numerator / denominator if denominator != 0 else 0.0
    _, p, _, _ = chi2_contingency(table, correction=False)

    return phi, p 

#*********************
# Kyle H will present on the topic of Fisher’s exact test, creating function named 'fisher_exact' 

def fisher_exact(table):
    '''
    Performs Fisher's exact test for a 2x2 contingency table.
    
    Parameters:
    table : array-like
        A 2x2 contingency table in the form of a list of lists or numpy array
        [[a, b], [c, d]] where:
        a = number of successes in group 1
        b = number of successes in group 2
        c = number of failures in group 1
        d = number of failures in group 2
    
    Returns:
    p_value : float
        The p-value from Fisher's exact test
    
    Example:
    >>> table = [[8, 2], [1, 5]]  # Treatment vs. Control recovery data
    >>> p_value = fisher_exact(table)
    >>> print(p_value)
    0.0350877192982456
    
    Notes:
    This function calculates the exact probability of observing the given table
    or one more extreme under the null hypothesis of independence.
    It is particularly useful for small sample sizes where chi-square 
    approximations may not be valid.
    '''
    import numpy as np
    from scipy import stats
    
    # Convert input to numpy array if it's not already
    table = np.array(table, dtype=np.int64)
    
    # Check if the table is 2x2
    if table.shape != (2, 2):
        raise ValueError("Fisher's exact test requires a 2x2 contingency table")
    
    # Perform Fisher's exact test using SciPy
    odds_ratio, p_value = stats.fisher_exact(table)
    
    return p_value

#*********************
# Emma H will present on the topic of Spearman correlation test, creating function named 'spearmanr' 
def spearmanr(x, y):

    '''
    Perform a Spearman rank-order correlation test.
    Parameters:
    x : array_like
        First sample.
    y : array_like
        Second sample.
    Returns:
    correlation : float
        The Spearman correlation.
    p_value : float
        The two-tailed p-value.

    Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 3, 4, 5, 6]
    >>> correlation, p_value = spearmanr(x, y)
    >>> print(correlation)
    1.0
    >>> print(p_value)
    0.0
    
    Notes:
    This function calculates the Spearman rank-order correlation coefficient and its p-value.
    The p-value is calculated using the two-tailed test.
    The function assumes that the input data is one-dimensional and of equal length.
    '''

    # Calculate the ranks of the data
    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(y)
    # Calculate the Spearman correlation
    correlation = np.corrcoef(ranks_x, ranks_y)[0, 1]
    # Calculate the p-value
    n = len(x)
    t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
    return correlation, p_value
#*********************
# Elijah J will present on the topic of Bartlett’s test, creating function named 'bartlett' 
def bartlett(*samples):
    '''
    Perform Bartlett's test for homogeneity of variances across multiple samples.
    
    Parameters:
    *samples : array_like
        Two or more arrays of sample data. Each array must have at least 2 observations.
        
    Returns:
    test_statistic : float
        The Bartlett test statistic (T).
    p_value : float
        The p-value for the test.
        
    Example:
    >>> sample1 = [1, 2, 3, 4, 5]
    >>> sample2 = [2, 3, 4, 5, 6]
    >>> sample3 = [3, 4, 5, 6, 7]
    >>> T, p = bartlett_test(sample1, sample2, sample3)
    >>> print(f"Test statistic: {T:.4f}, p-value: {p:.4f}")
    
    Notes:
    - Bartlett's test assumes normality. Use Levene's test if normality is questionable.
    - Requires at least two samples, each with ≥2 observations.
    '''
    samples = [np.asarray(sample) for sample in samples]
    k = len(samples)
    if k < 2:
        raise ValueError("At least two samples are required.")
    
    ni = np.array([len(sample) for sample in samples])
    if np.any(ni < 2):
        raise ValueError("Each sample must have at least 2 observations.")
    
    n = np.sum(ni)
    variances = np.array([np.var(sample, ddof=1) for sample in samples])
    
    # Pooled variance
    pooled_var = np.sum((ni - 1) * variances) / (n - k)
    
    # Numerator and denominator for the test statistic
    numerator = (n - k) * np.log(pooled_var) - np.sum((ni - 1) * np.log(variances))
    denominator = 1 + (np.sum(1 / (ni - 1)) - 1 / (n - k)) / (3 * (k - 1))
    
    T = numerator / denominator
    df = k - 1  # Correct degrees of freedom
    p_value = 1 - stats.chi2.cdf(T, df)
    
    return T, p_value
#*********************
# Gabriela J will present on the topic of Chi-square test of independence, creating function named 'chi2indep' 

#*********************
# Addison L will present on the topic of Permutation test for correlation, creating function named 'permutation_corr' 

#*********************
# Matthew L will present on the topic of Lilliefors test, creating function named 'lilliefors' 
#!pip install statsmodels
from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors

def lilliefors(data):
    '''
    Perform the Lilliefors test for normality.
    
    Parameters:
    data : array_like
        The dataset to test for normality.
        
    Returns:
    stat : float
        The test statistic (maximum distance between ECDF and CDF).
    p_value : float
        The p-value from the Lilliefors test.
    
    Example:
    >>> data = [2.1, 2.3, 1.9, 2.5, 2.0]
    >>> stat, p_value = lilliefors(data)
    >>> print(stat)
    >>> print(p_value)
    
    Notes:
    The test is similar to the Kolmogorov-Smirnov test but allows estimating
    the mean and variance from the data. It is suitable when population
    parameters are unknown.
    '''
    stat, p_value = sm_lilliefors(data)
    return stat, p_value

#*********************
# Hassan M will present on the topic of Partial correlation test, creating function named 'partial_corr' 

#*********************
# Bella P will present on the topic of Durbin-Watson test, creating function named 'durbin_watson' 
import statsmodels.api as sm
from scipy.stats import chi2
import numpy as np


def durbin_watson_test(residuals, n_params):
   """
   Calculates the Durbin-Watson statistic and its p-value.


   Args:
       residuals (array-like): Residuals from a regression model.
       n_params (int): Number of parameters in the regression model, including the intercept.


   Returns:
       tuple: (Durbin-Watson statistic, p-value)


   # Example Usage:
   residuals = [0.5, -0.2, 0.1, -0.3, 0.4]  # Example residuals
   n_params = 2  # Assume 1 predictor + intercept
   dw_stat, p_val = durbin_watson_test(residuals, n_params)
   print(f"Durbin-Watson statistic: {dw_stat:.4f}")
   print(f"Approximate p-value: {p_val:.4f}")
   Durbin-Watson statistic: 2.3000
   Approximate p-value: 0.2345


   Notes:
   - The Durbin-Watson statistic tests for autocorrelation in regression residuals.
   - A statistic near 2 suggests no autocorrelation.
   - Values closer to 0 suggest positive autocorrelation.
   - Values closer to 4 suggest negative autocorrelation.
   - The p-value is calculated using the chi-squared distribution, which give an approximation.
   - More accurate p-values can be obtained using a specific Durbin-Watson table.
   """
   # Compute Durbin-Watson statistic
   dw = sm.stats.durbin_watson(residuals)
   n = len(residuals)


   # Approximate p-value using the chi-squared distribution
   if dw <= 2:
       p_value = chi2.cdf(n * (2 - dw) / 2, n_params - 1)
   else:
       p_value = chi2.cdf(n * (dw - 2) / 2, n_params - 1)
  
   return dw, p_value

#*********************
# Chris RT will present on the topic of Fligner-Killeen test, creating function named 'fligner_killeen' 

#*********************
# Mariana S will present on the topic of Point-Biserial correlation test, creating function named 'pointbiserialr' 

#*********************
# Dylan S will present on the topic of Barnard’s exact test, creating function named 'barnard_exact'

from statsmodels.stats.contingency_tables import Table2x2

def barnard_exact(table):
    '''
    Perform Barnard’s Exact Test on a 2x2 contingency table.

    Parameters:
    table : array_like
        A 2x2 table, such as [[a, b], [c, d]]

    Returns:
    p_value : float
        The p-value from Barnard’s Exact Test
    '''
    tbl = Table2x2(table)
    result = tbl.test_nominal_association(method="barnard")
    return result.pvalue

#*********************
# Jacob S will present on the topic of Pearson correlation test, crating function named 'pearsonr' 

def pearson_correlation_test(x, y):
    '''
    Perform a Pearson correlation test between two arrays of continuous variables.
    
    Parameters:
    x : array_like
        First variable to correlate, must be continuous data.
    y : array_like
        Second variable to correlate, must be continuous data.
        
    Returns:
    r : float
        Pearson correlation coefficient between -1 and 1.
    p_value : float
        Two-tailed p-value for testing non-correlation.
    
    Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 3, 5, 6, 8]
    >>> r, p_value = pearson_correlation_test(x, y)
    >>> print(r)
    0.9914860303065454
    >>> print(p_value)
    0.000898315196118051
    
    Notes:
    - The Pearson correlation measures linear relationships only.
    - Values close to 1 indicate strong positive correlation.
    - Values close to -1 indicate strong negative correlation.
    - Values close to 0 indicate little to no linear correlation.
    - Both variables should follow approximately normal distributions for the p-value to be valid.
    - Requires at least 3 paired data points to compute.
    '''
    import scipy.stats as stats
    import numpy as np
    
    # Convert inputs to numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Check for valid inputs
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    if len(x) < 3:
        raise ValueError("Need at least 3 data points to compute correlation")
    
    # Calculate Pearson correlation and p-value
    r, p_value = stats.pearsonr(x, y)
    
    return r, p_value

#*********************
# Hayley S will present on the topic of Brown-Forsythe test, creating function named 'brown_forsythe' 

def brown_forsythe_test(data):
    # Step 1: Compute the transformed response variable z_ij
    z = [np.abs(group - np.median(group)) for group in data]

    # Step 2: Perform one-way ANOVA on z_ij
    F_stat, p_value = stats.f_oneway(*z)\
    
    if p_value < 0.05:
        print("Reject the null hypothesis: The variances are not equal.")

    return F_stat, p_value

#*********************
# Kayla T will present on the topic of Distance correlation test, creating function named 'dcor' 

#*********************
# Vivia VDM will present on the topic of White test, creating function named 'white_test' 

#*********************
# Fig V will present on the topic of Linear regression coefficient test, creating function named 'linreg' 

def linreg(X, y, coef_index = 0): 
    """"
    Performs a paired two sided t-test for a coefficient in linear regression
    The inputs are X, y, and coef_index
    X is a matrix of independent variables (including intercept)
    y is a vector of dependent variables, also known as the response variable
    coef_index is the index of the coefficient we are testing
    The coef_index begins at zero for the position element in the coefficient vector

    The function returns the t-statistic and p-value for the coefficient
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from scipy import stats

    # Fit the linear regression model
    model = LinearRegression().fit(X, y)

    # Calculate predictions & residuals 
    y_pred = model.predict(X)
    residuals = y - y_pred
    n = len(y)
    p = X.shape[1]  # number of predictors (including intercept)

    # Calculate the standard error of the coefficients 
    sigma_sq = np.sum(residuals**2) / (n-p-1)
    X_w_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept to X
    cov_matrix = sigma_sq * np.linalg.inv(np.dot(X_w_intercept.T, X_w_intercept))
    se = np.sqrt(np.diag(cov_matrix))

    # Calculate the t-statistic for the coefficient at coef_index
    t_stat = model.coef_[coef_index] / se[coef_index]
    # Calculate the p-value for the two-sided test  
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-p-1))
    

    return t_stat, p_value


#*********************
# AbbyMae W will present on the topic of Kendall’s Tau correlation test, creating function named 'kendalltau' 
import numpy as np
from scipy import stats
def kendalltau(x, y):
    '''
    This function calculates the Kendall's Tau correlation coefficient and its p-value.
    
    Parameters:
    x : First array
    y : Second array 
       
    Returns:
    tau : float
        The calculated Kendall's Tau statistic.
    p_value : float
        The two-tailed p-value.
    '''
    tau, p_value = stats.kendalltau(x, y)
    return tau, p_value
#*********************
# Alvina Y will present on the topic of F-test for overall regression, creating function named 'f_test' 
def ftest(y_true, y_pred, n_predictors):
    '''
    Perform an F-test for overall significance of regression.
    
    Parameters:
    y_true : array_like
        Observed/true dependent variable values.
    y_pred : array_like
        Predicted values from regression model.
    n_predictors : int
        Number of predictor variables in the model (excluding intercept).
        
    Returns:
    f_stat : float
        The calculated F-statistic.
    p_value : float
        The p-value for the F-test.  
        
    Example:
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
    >>> n_predictors = 2
    >>> f_stat, p_value = ftest(y_true, y_pred, n_predictors)
    >>> print(f_stat)
    185.71428571428572
    >>> print(p_value)
    0.000583853374492769
    
    Notes:
    - Tests H0: All regression coefficients (except intercept) are zero.
    - Uses scipy.stats.f.sf for p-value calculation.
    '''
    import numpy as np
    from scipy.stats import f
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_obs = len(y_true)
    
    # Calculate sums of squares
    ssr = np.sum((y_pred - np.mean(y_true))**2)  # Regression SS
    sse = np.sum((y_true - y_pred)**2)           # Error SS
    
    # Degrees of freedom
    df_reg = n_predictors
    df_resid = n_obs - n_predictors - 1
    
    # F-statistic
    f_stat = (ssr / df_reg) / (sse / df_resid)
    
    # P-value (right-tailed F-test)
    p_value = f.sf(f_stat, df_reg, df_resid)
    
    return f_stat, p_value
#*********************
# Polina Z will present on the topic of Harvey-Collier test, creating function named 'harvey_collier'

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_harvey_collier

def harvey_collier(X, y):
    """
    Performs the Harvey-Collier test for linearity in a linear regression model.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Predictor variables
    y : array-like, shape (n_samples,)
        Response variable

    Returns:
    p_value: float
        p-value of the Harvey-Collier test.
    
    Example:
    >>> X = np.linspace(0, 10, 100)
    >>> y = np.sin(X) + np.random.normal(scale=0.1, size=100)
    >>> p_value = harvey_collier(X, y)
    >>> print(p_value)

    Notes:
    The Harvey-Collier Test checks whether a linear model is appropriately specified as linear.
    A small p-value indicates that the model may be misspecified, and should be used with a nonlinear model instead.
    """

    # Ensure NumPy array. Converts data to 2D if it isn't already
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = sm.add_constant(X)

    # Fit linear regression model
    model = sm.OLS(y, X).fit()

    # Perform Harvey-Collier test and return p-value
    t_stat, p_value = linear_harvey_collier(model)
    return p_value

#*********************

