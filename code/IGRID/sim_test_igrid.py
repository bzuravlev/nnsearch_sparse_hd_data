from range import Range
import json
import scipy.sparse as sp

def slice_dimension(dimension_data, num_areas):
    
    return

def main():
    with open("../config.json") as c:
        config = json.load(c)
        db_path = config["db_path"]
        print(db_path)
        DB = sp.load_npz(db_path).tolil()
    return

if __name__=="__main__":
    main()