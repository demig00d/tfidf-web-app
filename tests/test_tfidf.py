from typing import Dict
from tfidf import tfidf
import numpy as np
import numpy.testing as npt

def analyse_file(filename: str) -> Dict[str, Dict[str, np.float64]]:
    with open(filename) as f:
        result_rows = tfidf([f.read()])
    return { 
        row["слово"]: 
            {
                "tf": row["tf"], 
                "idf": row["idf"],
            } for row in result_rows 
        }


def test_tfidf():
    result = analyse_file("./tests/data/doc1.txt")
    
    _ = result["air"]        is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["quality"]    is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["sunny"]      is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["island"]     is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["improved"]   is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["gradually"]  is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["throughout"] is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 
    _ = result["wednesday"]  is {"tf": np.float64(0.1), "idf": np.float64(1.0)} 

    result = analyse_file("./tests/data/doc2.txt")

    _ = result["air"]       is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["quality"]   is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["island"]    is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["singapore"] is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["continued"] is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["haze"]      is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 
    _ = result["wednesday"] is {"tf": np.float64(0.06666666666666668), "idf": np.float64(1.0)} 

    result = analyse_file("./tests/data/doc3.txt")
    	
    _ = result["air"]        is {"tf": np.float64(0.09523809523809525), "idf": np.float64(1.0)} 
    _ = result["different"]  is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)} 
    _ = result["island"]     is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)}
    _ = result["located"]    is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)}
    _ = result["monitored"]  is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)}
    _ = result["monitoring"] is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)}
    _ = result["network"]    is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)}
    _ = result["quality"]    is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)} 
    _ = result["singapore"]  is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)} 
    _ = result["stations"]   is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)} 
    _ = result["through"]    is {"tf": np.float64(0.04761904761904762), "idf": np.float64(1.0)} 

    result = analyse_file("./tests/data/doc4.txt")
    	
    _ = result["air"]       is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
    _ = result["got"]       is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
    _ = result["quality"]   is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
    _ = result["singapore"] is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
    _ = result["worse"]     is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
    _ = result["wednesday"] is {"tf": np.float64(0.1111111111111111), "idf": np.float64(1.0)} 
