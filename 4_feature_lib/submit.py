import featurelib as fl

import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime
import functools

from typing import List, Union, Optional, Dict

import sklearn.base as skbase

# from feature_impl import dask_groupby

from category_encoders.leave_one_out import LeaveOneOutEncoder


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer): # сколько уникальных категорий покупал клиеент за последние n дней (delta)
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        
        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        receipts = receipts.loc[date_mask]
        
        receipts['transaction_dow'] = receipts['transaction_datetime'].dt.dayofweek
        
        receipts['ones'] = 1

        receipts = receipts.categorize(columns=['transaction_dow'])

        result = dd.pivot_table(receipts, index='client_id', values='ones',
                                    columns='transaction_dow', aggfunc='sum')

        result = result.fillna(0)
        
        for day in result.columns:
            result = result.rename(columns={day: f'purchases_count_dw{day}__{self.delta}d'})
            
#         result.columns = result.columns.astype(str)
        result = result.reset_index()
    
        result.columns.name = 'day_of_week'

        return result

# register_calcer(UniqueCategoriesCalcer)


class FavouriteStoreCalcer(fl.DateFeatureCalcer): # сколько уникальных категорий покупал клиеент за последние n дней (delta)
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        
        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        receipts = receipts.loc[date_mask]
        
        result = receipts.groupby(by=['client_id', 'store_id']).size().reset_index()
        
        result = result.map_partitions(lambda x: x.sort_values([0, 'store_id'], ascending=[False, False]))
        
        result = result.drop_duplicates('client_id', keep='first')
        
        result = result.rename(columns={'store_id': f"favourite_store_id__{self.delta}d"})
        
        result = result.drop(0, axis=1)
        
        return result

# register_calcer(UniqueCategoriesCalcer)


class ExpressionTransformer(skbase.BaseEstimator, skbase.TransformerMixin):

    # expression: str # “регулярное” выражение для расчета признака. (пример см. ниже)
    # col_result: str, # название колонки, в которой будет сохранен результат
    
    def __init__(self, expression: str, col_result: str, **kwargs):
        self.expression = expression
        self.col_result = col_result
        super().__init__(**kwargs)
    
    # Метод fit (пустой). Ничего не делает. Возвращает сам объект.
    def fit(self, *args, **kwargs):
        return self

    # Описание результата расчета
    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
#         s = "({d}['purchases_count_dw5__60d'] + {d}['purchases_count_dw6__60d'])"
        s = self.expression.replace('{d}', 'data')
        s = s.replace('[', '.')
        s = s.replace(']', '')
        s = s.replace("'", "")
        data[self.col_result]  = pd.eval(s)
        
        return data
    
    
class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    # expression: str # “регулярное” выражение для расчета признака. (пример см. ниже)
    # col_result: str, # название колонки, в которой будет сохранен результат
    
    def __init__(self, col_categorical: str, col_target: str, col_result: str, **kwargs):
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result
        super().__init__(**kwargs)
    
    def fit(self, data: pd.DataFrame, *args, **kwargs):
        enc = LeaveOneOutEncoder()
        X = data[self.col_categorical].astype(str)
        y = data[self.col_target]
        enc.fit(np.asarray(X), y)
            
        self.enc = enc
#         self.X = X
        
        return self

    # Описание результата расчета
    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        
        if self.col_target not in data.columns:
            X = data[self.col_categorical].astype(str)
            data[self.col_result] = self.enc.transform(np.asarray(X))
            
        else:
            enc = LeaveOneOutEncoder()
            X = data[self.col_categorical].astype(str)
            y = data[self.col_target]
            enc.fit(np.asarray(X), y)
            data[self.col_result] = enc.transform(np.asarray(X))
        
        return data
        