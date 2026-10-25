def feature_engineering(df):
    result = df.copy()
    
    result['BILL_AMT_SUM'] = result[[f'BILL_AMT{i}' for i in range(1, 7)]].sum(axis=1)
    result['PAY_AMT_SUM'] = result[[f'PAY_AMT{i}' for i in range(1, 7)]].sum(axis=1)
    result['BILL_AMT_MEAN'] = result[[f'BILL_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    result['PAY_AMT_MEAN'] = result[[f'PAY_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    
   
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    result['NUM_DELAYED_PAYMENTS'] = (result[pay_cols] > 0).sum(axis=1)
    limit = result['LIMIT_BAL'].where(result['LIMIT_BAL'] > 0)
    result['BILL_TO_LIMIT_RATIO'] = (result['BILL_AMT_SUM'] / limit)
    return result


