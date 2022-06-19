import pandas as pd


def make_dict():
    csv_to_file = {}
    file_to_csv = {}
    a_file = open("moshuir_data/all_files.txt")
    file_contents = a_file.read()
    contents_split = file_contents.splitlines()

    for el in contents_split:
        parts = el.split()
        csv_to_file[parts[1]] = parts[0]
        file_to_csv[parts[0]] = parts[1]

    return csv_to_file, file_to_csv


def read_csv():
    data_frame = pd.read_csv('moshuir_data/four_choice_response.csv', index_col=0)

    csv_to_file, file_to_csv = make_dict()
    chosen = dict.fromkeys(file_to_csv.keys(), 0.0)
    cols = data_frame.columns
    for i in range(1, len(cols)):
        column = cols[i]
        for row in data_frame[column]:
            if 'chosen' in column:
                curr = chosen[csv_to_file[row]]
                chosen.update({csv_to_file[row]: curr + 1})
            else:
                continue
    chosen.update((x, y / len(csv_to_file)) for x, y in chosen.items())
    return chosen
