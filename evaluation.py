import pandas as pd

def evaluate(results, train, test, user):
    # get the test split for the chosen user
    user_test = test.loc[test['reviewerID'] == user]
    user_test_relev = user_test[["asin", "overall"]].copy()

    # rename overall to actual
    user_test_relev = user_test_relev.rename(columns={"overall": "actual"})

    # set asin as index for both
    user_test_relev = user_test_relev.set_index('asin')
    results = results.set_index('asin')

    print()

    comparison = user_test_relev.join(results)
    print(comparison)

    # for each asin in the test split, store the predicted score 
    #   from the results with the actual score [predicted, actual]

    exit()