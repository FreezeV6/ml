import os
import math
import random

def read_raw(path: str = None) -> str:
    try:
        with open(file=path, mode='r', encoding='utf-8') as data:
            return data.read().lstrip('\ufeff').strip('"')
    except FileNotFoundError:
        print(f'\033[31mBrak pliku!\n\033[33mMożliwy błąd w ścieżce: {path}\n\033[0m')
        return ''

def delimiter_sniffer(raw_data: str = None) -> str:
    possible_delimiters = ['\t', ';', ',', ' ']
    sample = [line for line in raw_data.splitlines() if line.strip()][:5]
    delimiter_count = {}
    try:
        for sep in possible_delimiters:
            counts = [row.count(sep) for row in sample]
            if counts and all(c == counts[0] and c > 0 for c in counts):
                delimiter_count[sep] = counts[0]
        if delimiter_count:
            return max(delimiter_count, key=delimiter_count.get)
        print('\033[33mNieznany separator, zastosowany domyślny: ","\033[0m')
        return ','
    except StopIteration or RuntimeError:
        print('\033[33mBłąd procesowania danych!\n\033[0m')
        return ''

def parse_rows(raw_data: str, sep: str) -> list:
    return [line.split(sep) for line in raw_data.splitlines() if line.strip()]

def extract_features(data: dict = None) -> dict:
    value_count = {}
    for column_name, value_list in data.items():
        count = {}
        for v in value_list:
            count[v] = count.get(v, 0) + 1
        value_count[column_name] = count
    return value_count

def calculate_probabilities(features: dict) -> dict:
    probabilities = {}
    for column_name, value_count_dict in features.items():
        total_count = sum(value_count_dict.values())
        probabilities[column_name] = {value: count/total_count for value, count in value_count_dict.items()}
    return probabilities

def calculate_entropy(probabilities: dict) -> dict:
    entropies = {}
    for column_name, probabilities_dict in probabilities.items():
        entropy = 0.0
        for prob in probabilities_dict.values():
            if prob >= 0:
                entropy -= prob * math.log2(prob)
        entropies[column_name] = entropy
    return entropies

def calculate_information_function(data: dict, decision: str) -> dict:
    total_count = len(data[decision])
    info_func_dict = {}
    for attribute, values_list in data.items():
        if attribute == decision:
            continue
        groups = {}
        for index, value in enumerate(values_list):
            groups.setdefault(value, []).append(index)
        subset_info_func = 0.0
        for value_indexes in groups.values():
            subset_decision_value_counts = {}
            for value_index in value_indexes:
                decision_value = data[decision][value_index]
                subset_decision_value_counts[decision_value] = subset_decision_value_counts.get(decision_value, 0) + 1
            probabilities = calculate_probabilities({ decision: subset_decision_value_counts })
            decision_entropy = calculate_entropy(probabilities)[decision]
            subset_info_func += (len(value_indexes) / total_count) * decision_entropy
        info_func_dict[attribute] = subset_info_func
    return info_func_dict

def calculate_information_gain(entropies: dict, info_func: dict, decision: str) -> dict:
    decision_entropy_value = entropies.get(decision, 0)
    return {attribute: decision_entropy_value - info_func[attribute] for attribute in info_func}

def calculate_gain_ratio(data: dict, decision: str) -> dict:
    features = extract_features(data)
    probs = calculate_probabilities(features)
    entropies = calculate_entropy(probs)
    cond_ent = calculate_information_function(data, decision)
    info_gains = calculate_information_gain(entropies, cond_ent, decision)
    gain_ratio = {}
    for attribute, info_gain in info_gains.items():
        attr_entropy = entropies.get(attribute, 0.0)
        if attr_entropy > 0:
            gain_ratio[attribute] = info_gain / attr_entropy
        else:
            gain_ratio[attribute] = 0.0
    return gain_ratio

def make_leaf(data: dict, decision: str, depth: int) -> dict:
    decision_distribution = extract_features({decision: data[decision]})[decision]
    most_common_label = max(decision_distribution, key=decision_distribution.get)
    return {'type': 'leaf', 'label': most_common_label, 'count': len(data[decision]), 'depth': depth}

def build_decision_tree(data: dict, attributes: list, decision: str, depth: int = 0) -> dict:
    ratios = calculate_gain_ratio(data, decision)
    valid_gains = [(a, ratios[a]) for a in attributes if a in ratios]
    if not valid_gains:
        return make_leaf(data, decision, depth)
    best_attribute, best_gain = max(valid_gains, key=lambda x: x[1])
    if best_gain <= 0:
        return make_leaf(data, decision, depth)
    node = {
        'type': 'node',
        'attribute': best_attribute,
        'info_gain': round(best_gain, 6),
        'branches': {},
        'depth': depth
    }
    for branch_value in set(data[best_attribute]):
        record_indexes = [ri for ri, rv in enumerate(data[best_attribute]) if rv == branch_value]
        subset = {column_name: [data[column_name][i] for i in record_indexes] for column_name in data}
        node['branches'][branch_value] = build_decision_tree(
            subset,
            [a for a in attributes if a != best_attribute],
            decision,
            depth + 1
        )
    return node

def print_tree(tree: dict, indent: str = '') -> str:
    line = ''
    if tree['type'] == 'leaf':
        line += f"Decyzja: {tree['label']}"
    else:
        line += f"Atrybut = {tree['attribute']}"
    if tree['type'] == 'node':
        for branch_val, subtree in tree['branches'].items():
            line += f"\n{indent}    {branch_val} -> "
            line += print_tree(subtree, indent + '    ')
    return line

def classify_subtree(tree: dict, record: dict):
    """
    Rekurencyjnie klasyfikuje pojedynczy rekord (słownik {atrybut: wartość})
    za pomocą zbudowanego drzewa decyzyjnego.
    """
    if tree['type'] == 'leaf':
        return tree['label']
    attr = tree['attribute']
    val = record.get(attr)
    if val in tree['branches']:
        return classify_subtree(tree['branches'][val], record)
    else:
        # Jeśli wartość nie występowała podczas treningu, zwracamy etykietę najczęstszej gałęzi:
        # wybieramy gałąź z największą liczbą liści lub po prostu etykietę większościową
        # Tutaj uproszczenie: zwróć etykietę "najczęstszej" gałęzi według liczebności poddrzewa
        leaf_counts = {}
        def count_leaves(subt):
            if subt['type'] == 'leaf':
                return {subt['label']: 1}
            total = {}
            for br in subt['branches'].values():
                c = count_leaves(br)
                for k, v in c.items():
                    total[k] = total.get(k, 0) + v
            return total
        for br_val, subtree in tree['branches'].items():
            cnts = count_leaves(subtree)
            for label, cnt in cnts.items():
                leaf_counts[label] = leaf_counts.get(label, 0) + cnt
        if leaf_counts:
            return max(leaf_counts, key=leaf_counts.get)
        else:
            # Jeśli nie można wyznaczyć, zwróć None
            return None

def evaluate_classification(true_labels: list, pred_labels: list):
    """
    Oblicza accuracy, precision i recall ze zbioru prawdziwych i przewidzianych etykiet.
    Zakładamy klasy binarne: zakładamy, że pozytywną klasą jest etykieta '1' (inaczej pierwsza występująca)
    """
    TP = FP = TN = FN = 0
    unique = set(true_labels)
    if '1' in unique:
        positive = '1'
    else:
        positive = next(iter(unique))
    for t, p in zip(true_labels, pred_labels):
        if p is None:
            if t == positive:
                FN += 1
            else:
                FP += 1
            continue
        if t == positive and p == positive:
            TP += 1
        elif t != positive and p == positive:
            FP += 1
        elif t == positive and p != positive:
            FN += 1
        else:
            TN += 1
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return accuracy, precision, recall

def prune_tree(node: dict, validation_data: dict, decision: str):
    """
    Przycinanie drzewa decyzyjnego na podstawie zbioru walidacyjnego.
    Używa reguły:
        e_leaf <= e_subtree + sqrt( e_subtree * (1 - e_subtree) / |P| )
    gdzie e_subtree to błąd drzewa na zbiorze walidacyjnym docierającym do danego węzła,
    e_leaf – błąd gdy zastąpimy węzeł liściem prognozującym większościową etykietę (według walidacji).
    """
    # Jeśli to liść, nic nie robimy
    if node['type'] == 'leaf':
        return

    attr = node['attribute']
    for branch_val, subtree in list(node['branches'].items()):
        subset_indices = [i for i, v in enumerate(validation_data[attr]) if v == branch_val]
        if not subset_indices:
            continue
        child_validation = {col: [validation_data[col][i] for i in subset_indices]
                            for col in validation_data}
        prune_tree(subtree, child_validation, decision)

    P = len(validation_data[decision])
    if P == 0:
        return

    true_labels = []
    pred_labels = []
    for i in range(P):
        record = {col: validation_data[col][i] for col in validation_data}
        true_labels.append(validation_data[decision][i])
        pred_labels.append(classify_subtree(node, record))

    errors_subtree = sum(1 for t, p in zip(true_labels, pred_labels) if t != p)
    e_sub = errors_subtree / P

    counts = {}
    for lab in true_labels:
        counts[lab] = counts.get(lab, 0) + 1
    majority_label = max(counts, key=counts.get)
    # Oblicz błąd liścia (e_leaf)
    errors_leaf = P - counts[majority_label]
    e_leaf = errors_leaf / P

    # Reguła przycinania
    threshold = e_sub + math.sqrt(e_sub * (1 - e_sub) / P) if P > 0 else e_sub
    if e_leaf <= threshold:
        # Przycinamy: zastępujemy węzeł liściem
        node.clear()
        node['type'] = 'leaf'
        node['label'] = majority_label
        node['count'] = P
        # depth pozostawiamy bez zmian (nie trzeba go dalej używać)
        node['depth'] = node.get('depth', 0)

def train_and_test(data: dict, attributes: list, decision: str, test_frac: float = 0.3):
    """
    Dzieli dane na zbiory: treningowy i testowy według podanego ułamka.
    Buduje drzewo na zbiorze treningowym, a następnie testuje na zbiorze
    testowym, obliczając accuracy, precision i recall.
    """
    # Liczba przykładów ogółem
    N = len(data[decision])
    indices = list(range(N))
    random.shuffle(indices)

    n_test = int(N * test_frac)
    n_train = N - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Funkcja pomocnicza do tworzenia podzbioru
    def subset(data_dict, idx_list):
        return {col: [data_dict[col][i] for i in idx_list] for col in data_dict}

    train_data = subset(data, train_idx)
    test_data = subset(data, test_idx) if n_test > 0 else {col: [] for col in data}

    # Budowanie drzewa na zbiorze treningowym
    tree = build_decision_tree(train_data, attributes, decision)
    print("\n=== DRZEWO ZBUDOWANE NA ZBIORZE TRENINGOWYM ===")
    print(print_tree(tree))

    # Testowanie na zbiorze testowym
    P_test = len(test_data[decision])
    true_test = []
    pred_test = []
    for i in range(P_test):
        record = {col: test_data[col][i] for col in test_data}
        true_test.append(test_data[decision][i])
        pred_test.append(classify_subtree(tree, record))

    accuracy, precision, recall = evaluate_classification(true_test, pred_test)
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    print(f"Dokładność (accuracy): {accuracy:.4f}")
    print(f"Precyzja (precision): {precision:.4f}")
    print(f"Czułość (recall): {recall:.4f}")

    return tree, (accuracy, precision, recall)

def main():
    path = input("Podaj ścieżkę do pliku: ")
    raw = read_raw(path)
    if not raw:
        return
    sep = delimiter_sniffer(raw)
    rows = parse_rows(raw, sep)

    print("Potencjalne nagłówki z pierwszej linii:")
    for i, h in enumerate(rows[0], 1):
        print(f"  {i}. {h}")
    headers_input = input("Czy są nagłówki? (tak/nie): ").strip().lower()
    if headers_input.startswith('t'):
        headers = rows[0]
        data_rows = rows[1:]
    else:
        cols = len(rows[0])
        headers = [f"c{i+1}" for i in range(cols)]
        data_rows = rows

    print("Wybierz kolumnę decyzyjną:")
    for i, h in enumerate(headers, 1):
        print(f"  {i}. {h}")
    sel = int(input("Numer kolumny: ")) - 1
    headers[sel] = 'd'

    data = {h: [r[i] for r in data_rows] for i, h in enumerate(headers)}
    attributes = [h for h in headers if h != 'd']

    try:
        test_pct = float(input("Podaj procent danych do zbioru testowego (0-100, domyślnie 30): ") or "30")
        test_frac = max(0.0, min(1.0, test_pct / 100.0))
    except ValueError:
        print("Błędne wartości procentowe. Używam domyślnej wartości: 30% test.")
        test_frac = 0.30

    # Uruchamiamy procedurę trenowania i testowania
    tree, metrics = train_and_test(data, attributes, 'd', test_frac=test_frac)

    os.makedirs('result', exist_ok=True)
    with open('result/breast-cancer.txt', 'w', encoding='utf-8') as f:
        f.write(print_tree(tree))
    print("Drzewo zapisano w result/breast-cancer.txt")

if __name__ == '__main__':
    main()

