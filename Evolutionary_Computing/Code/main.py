from algorithm import EA
from os import listdir


def train(selection, runs=10):
    for enemy in enemies:
        for i in range(runs):
            EA(selection=selection, enemy=enemy, run=i)


def test(selection, runs=5):
    final_results = []
    for enemy in enemies:
        gains = [EA(selection=selection, enemy=enemy, run=i, run_mode='test',
                    best_run=find_best_run(selection, enemy)) for i in range(runs)]
        final_results.append([f'{selection} vs Enemy{enemy}'] + gains)
    [print(result) for result in final_results]


def find_best_run(selection, enemy):
    print(len(listdir(f'train/Enemy {enemy}/{selection}')))
    return max(range(len(listdir(f'train/Enemy {enemy}/{selection}'))-1), key=lambda i:
    float(open(f'train/Enemy {enemy}/{selection}/train_{i}/results.txt').readlines()[-1].split()[1]))


enemies = [3, 5, 8]

# train('Ranked')
# test('Elitism')
