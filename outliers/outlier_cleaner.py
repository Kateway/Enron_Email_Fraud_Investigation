#!/usr/bin/python

import numpy as np
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    size = len(predictions)
    errors = np.empty(size)

    # Calculate errors.  It should be the absolute value.
    for record in range(size):
        errors[record] = abs(predictions[record] - net_worths[record])

    # Find the 10% percentile
    percentile_10 = np.percentile(errors, 90)

    # Keep record if the error is less than 10% percentile.
    for record in range(size):
        if errors[record] < percentile_10:
            cleaned_data.append((ages[record], net_worths[record], errors[record]))

    # print len(cleaned_data)
    
    return cleaned_data

