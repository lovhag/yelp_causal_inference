import ijson
import pandas as pd

def _data_entry_is_wanted(data_entry, wanted_values):
    is_wanted = True

    for key, value in wanted_values.items():
        is_wanted = is_wanted and (data_entry[key] in value)

    return is_wanted

def get_data_frame_from_file(filename, fields, **kwargs):
    # fields = ["user_id","business_id","stars","useful","text","date"]

    data_dict = {}
    for field in fields:
        data_dict[field] = []

    max_nbr_items = {}
    if 'max_nbr_items' in kwargs:
        max_nbr_items = kwargs['max_nbr_items']

    wanted_values = {}
    if 'wanted_values' in kwargs:
        wanted_values = kwargs['wanted_values']

    with open(filename, 'rb') as input_file:
            # load json iteratively
            parser = ijson.parse(input_file, multiple_values=True)
            current_nbr_items = 0
            for prefix, event, value in parser:
                #print("%s: %s" %(prefix,value))

                if event == 'start_map':
                    data_entry = {}
                elif event == 'end_map':
                    if (not wanted_values or 
                    (wanted_values and _data_entry_is_wanted(data_entry, wanted_values))):
                        # add entry to data
                        for key, value in data_entry.items():
                            data_dict[key].append(value)

                        current_nbr_items += 1
                        if max_nbr_items <= current_nbr_items :
                            break
                else:
                    if prefix in fields:
                        data_entry[prefix] = value
                

    return pd.DataFrame(data_dict)