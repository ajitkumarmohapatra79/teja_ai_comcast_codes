import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{"NUM_PORTIN":1,"THIRD_PARTY_ID_SCORE":413,"FIRST_PARTY_ID_SCORE":423,"MAKE1":"Apple","MONTHLYRECURRINGCHARGE":45.0,"HOUR_OF_DAY":23,"TOTAL_PRICE":0.0,"PRICE1":0.0,"ONETIMECHARGE":0.0,"IDA_RESULT":"GREEN","MODEL1":"iPhone 6 Plus","INSTALLMENT_AMOUNT":0.0,"IS_EXISTING_CUSTOMER":"N"}])
output_sample = np.array([0])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'orderholdaksV2')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        if result ==0:
            result = 'FAST'
            print('FAST')
        elif result ==1:
            result = 'ON TARGET'
            print('ON TARGET')
        else:
            result = 'SLOW'
            print('SLOW')
        print('result') 
        fraud_status =[]
        fraud_status.append(result)
        return fraud_status
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result": result.tolist()})
