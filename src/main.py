from math import log2

def delimiter_sniffer(raw_data: str = None) -> str:
    allowed_delimiters = ['\t', ';', ',', ' ']
    lines = raw_data.splitlines()[:5]
    delimiter_counts = {}
    for delimiter in allowed_delimiters:
        counts = []
        for line in lines:
            if line.strip():
                counts.append(line.count(delimiter))
        if counts and all(count == counts[0] and count > 0 for count in counts):
            delimiter_counts[delimiter] = counts[0]
    if delimiter_counts:
        detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    else:
        print('\033[33mNie udało się jednoznacznie określić separatora. Domyślnie użyto ",".\033[0m')
        detected_delimiter = ','
    return detected_delimiter


def read_data(path: str = None, cont_labels: bool = False) -> (str, str, list):
    get_data_type = path.split('.')[-1]
    try:
        with open(path, mode='r') as file:
            raw_data = file.read().strip('ď»ż').strip('\ufeff')
            first_line = raw_data.splitlines()[0]
            delimiter = delimiter_sniffer(raw_data)
            labels = first_line.split(delimiter) if cont_labels else \
                [f"c{i+1}" for i in range(len(first_line.split(delimiter))-1)] + ["d"]
            return raw_data, get_data_type, labels
    except FileNotFoundError:
        print(f'\033[31mBrak pliku!\n\033[33mMożliwy błąd w ścieżce: {path}\n\033[0m')
        return '', '', []

def to_array(raw_data: str = None, delimiter: str = None, data_type: str = None, cont_labels: bool = False) -> dict:
    if data_type == 'csv' or data_type == 'txt' or data_type == 'tsv':
        try:
            splitted_lines = (line.split(delimiter) for line in raw_data.splitlines())
            labels = next(splitted_lines) if cont_labels else [f"c{i+1}" for i in range(len(next(splitted_lines)) - 1)] + ["d"]
            return dict(zip(labels, zip(*splitted_lines)))
        except StopIteration and RuntimeError:
            print('\033[33mBłąd procesowania danych!\n\033[0m')
            return {}
    else:
        print('\033[33mNieznany format pliku!\033[0m')
        return {}

def extract_features(data: dict = None) -> dict:
    value_count = {}
    for column_name, value_tuple in data.items():
        count = {}
        for value in value_tuple:
            count[value] = count.get(value, 0) + 1
        value_count[column_name] = count
    return value_count

def calculate_probabilities(features: dict = None) -> dict[str, dict[str, float]]:
    probabilities = {}
    for column_name, value_count_dict in features.items():
        total_count = sum(value_count_dict.values())
        column_probs = {}
        for value, value_count in value_count_dict.items():
            column_probs[value] = value_count / total_count
        probabilities[column_name] = column_probs
    return probabilities

def calculate_entropy(probabilities: dict = None) -> dict:
    entropies = {}
    for column_name, prob_dict in probabilities.items():
        entropy = 0.0
        for value, p in prob_dict.items():
            if p >= 0:
                entropy += p * log2(p)
        entropies[column_name] = -entropy
    return entropies

def calculate_information_gain(entropies: dict, decision_attribute: str = 'd') -> dict:
    information_gains = {}
    if decision_attribute not in entropies:
        raise ValueError(f"\033[31mNie znaleziono kolumny decyzyjnej '{decision_attribute}' w danych!\033[0m")
    dataset_entropy = entropies[decision_attribute]
    for attribute, entropy in entropies.items():
        if attribute != decision_attribute:
            information_gains[attribute] = dataset_entropy - entropy
    return information_gains

def show_results(entropies: dict = None, info_gain: dict = None) -> None:
    for column_name, entropy_value in entropies.items():
        print(f"\033[36m------- \033[35m{column_name} \033[36m-------")
        print(f"\033[36mEntropia = \033[32m{entropy_value}\n")
    print("\033[33m=== Przyrost informacji ===\033[0m")
    for attr, gain in info_gain.items():
        print(f"\033[34m{attr}: \033[32m{gain:.4f}\033[0m")


def get_data_info_inputs() -> (str, str):
    path = input("Podaj ścieżkę do pliku danych: \n")
    if type(path) != str:
        raise TypeError("\033[33mNiepoprawna ścieżka!\033[0m")
    cont_labels = input("Czy dane zawierają nagłówki? (tak/nie) \n")
    if cont_labels.lower() == 'nie':
        cont_labels = False
    elif cont_labels.lower() == 'tak':
        cont_labels = True
    else:
        cont_labels = False
        print("Nie rozpoznano odpowiedzi, zakładam, że nie \n")
    return path, cont_labels

def get_decision_column(labels: list = None) -> str:
    decision_attribute = input("Podaj nazwę kolumny decyzyjnej lub jej indeks (np. 'salary' lub '4'): \n")
    if decision_attribute.isnumeric():
        decision_attribute = int(decision_attribute) - 1
        try:
            decision_attribute_name = labels[decision_attribute]
        except IndexError:
            raise ValueError(f"\033[31mIndeks kolumny decyzyjnej ({decision_attribute}) poza zakresem!\033[0m")
    else:
        decision_attribute_name = decision_attribute.lower()
        if decision_attribute_name not in labels:
            raise ValueError(f"\033[31mNie znaleziono kolumny decyzyjnej o nazwie '{decision_attribute_name}'!\033[0m")

    return decision_attribute_name

def main():
    # ścieżki testowe
    path1 = r'../sample_data/testGielda/gieldaLiczby.txt'
    path2 = r'../sample_data/testGielda/gielda.txt'
    path3 = r'../sample_data/ansc.csv'
    path4 = r'../sample_data/dino.tsv'
    path5 = r'../sample_data/classification.csv'
    #
    path, cont_labels = get_data_info_inputs()
    raw_data, data_type, labels = read_data(path, cont_labels=cont_labels)
    delimiter = delimiter_sniffer(raw_data)
    decision_attribute = get_decision_column(labels)
    arrayed = to_array(raw_data=raw_data, data_type=data_type, delimiter=delimiter, cont_labels=cont_labels)
    features = extract_features(data=arrayed)
    probabilities = calculate_probabilities(features=features)
    entropies = calculate_entropy(probabilities=probabilities)
    information_gains = calculate_information_gain(entropies=entropies, decision_attribute=decision_attribute)
    show_results(entropies=entropies, info_gain=information_gains)


if __name__ == '__main__':
    main()

