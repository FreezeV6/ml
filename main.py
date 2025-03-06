def log2(x: float, epsilon: float = 1e-12) -> float:
    if x <= 0:
        raise ValueError("\033[31mlog2: x musi być > 0\033[0m")
    if x == 1:
        return 0.0
    y_min, y_max = -25.0, 25.0
    base = 2.0
    while True:
        y_mid = (y_min + y_max) / 2
        val = base ** y_mid
        diff = val - x
        if abs(diff) < epsilon:
            nearest_int = round(y_mid)
            if abs(y_mid - nearest_int) < 1e-12:
                return float(nearest_int)
            return y_mid
        if diff < 0:
            y_min = y_mid
        else:
            y_max = y_mid

def read_data(path: str = None) -> str:
    try:
        with open(path, mode='r') as file:
            return file.read().strip('ď»ż').strip('\ufeff')
    except FileNotFoundError:
        print(f'\033[31mBrak pliku!\n\033[33mMożliwy błąd w ścieżce: {path}\n\033[0m')
        return ''

def to_array(raw_data: str = None, delimiter: str = None, data_type: str = None) -> dict:
    allowed_delimiters = ['\t', ';', ',', ' ']
    match data_type:
        case "csv":
            print('\033[33mNieznany separator, zastosowany domyślny: ","\033[0m') if delimiter not in allowed_delimiters else delimiter
            try:
                splitted_lines = (line.split(delimiter) for line in raw_data.splitlines())
                labels = next(splitted_lines)
                return dict(zip(labels, zip(*splitted_lines)))
            except StopIteration:
                print('\033[33mBłąd procesowania danych!\n\033[0m')
                return {}
        case _:
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
        total_entropy = 0.0
        for value, p in prob_dict.items():
            if p > 0:
                total_entropy -= p * log2(p)
        entropies[column_name] = total_entropy
    return entropies

def show_results(entropies: dict) -> None:
    for column_name, entropy_value in entropies.items():
        print(f"\033[36m------- \033[35m{column_name} \033[36m-------")
        print(f"\033[36mEntropia = \033[32m{entropy_value}\n")


def main():
    raw_data = read_data(r'sample_data/classification.csv')
    arrayed = to_array(raw_data=raw_data, data_type='csv', delimiter=',')
    features = extract_features(data=arrayed)
    probabilities = calculate_probabilities(features=features)
    entropies = calculate_entropy(probabilities=probabilities)
    show_results(entropies=entropies)

if __name__ == '__main__':
    main()
