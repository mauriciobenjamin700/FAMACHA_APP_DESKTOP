from pandas import DataFrame
from os.path import expanduser,join 

def save_results(df: DataFrame = None, mode:str = "csv")-> bool:
    
    result = False
    
    if df is not None:
        output_file = join(expanduser('~'),"Documents","result")
        match mode:
            case "csv":
                df.to_csv(output_file+".csv",index=False)
                result = True
            case "json":
                df.to_json(output_file+".json", orient='records')
                result = True
            case "excel":
                df.to_excel(output_file+".xlsx",index=False)
                result = True
            case _:
                pass
                
                
    return result