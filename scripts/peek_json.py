import ijson
import pprint

def analyze_json():
    with open(r'datasets/raw/mp_e_form.json', 'rb') as f:
        parser = ijson.parse(f)
        indices = []
        for prefix, event, value in parser:
            if prefix == 'index.item':
                indices.append(value)
            elif prefix == 'index' and event == 'end_array':
                break
        
        print(f"Total indices: {len(indices)}")
        if len(indices) > 0:
            print(f"First 5 indices: {indices[:5]}")
                
if __name__ == '__main__':
    analyze_json()
