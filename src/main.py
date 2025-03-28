from math import log2

def read_data(path: str = None) -> str:
    try:
        with open(path, mode='r') as file:
            return file.read().strip('ď»ż').strip('\ufeff')
    except FileNotFoundError:
        print(f'\033[31mBrak pliku!\n\033[33mMożliwy błąd w ścieżce: {path}\n\033[0m')
        return ''

def to_array(raw_data: str = None, delimiter: str = None, data_type: str = None, cont_labels: bool = False) -> dict:
    allowed_delimiters = ['\t', ';', ',', ' ']
    if data_type == 'csv':
        print('\033[33mNieznany separator, zastosowany domyślny: ","\033[0m') if delimiter not in allowed_delimiters else delimiter
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


def main():
    raw_data = read_data(r'../sample_data/testGielda/gielda.txt')
    arrayed = to_array(raw_data=raw_data, data_type='csv', delimiter=',', cont_labels=False)
    features = extract_features(data=arrayed)
    probabilities = calculate_probabilities(features=features)
    entropies = calculate_entropy(probabilities=probabilities)
    information_gains = calculate_information_gain(entropies=entropies, decision_attribute='d')
    show_results(entropies=entropies, info_gain=information_gains)


if __name__ == '__main__':
    main()

