import random
from typing import List

supported_ops = ['+', '-', '*']


def f(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    else:
        return a * b


def generate(ops: List[str], min, max, num, save_path):
    with open(save_path, 'w', encoding='utf-8') as fw:
        fw.write('input\ttarget\n')
        for i in range(0, num):
            op = random.choice(ops)
            assert op in supported_ops
            a = random.randint(min, max)
            b = random.randint(min, max)
            c = f(a, b, op)
            fw.write('%d%s%d=\t%d' % (a, op, b, c) + '\n')


def generate_test(ops: List[str], min, max, num, save_path):
    with open(save_path, 'w', encoding='utf-8') as fw:
        fw.write('input\n')
        for i in range(0, num):
            op = random.choice(ops)
            assert op in supported_ops
            a = random.randint(min, max)
            b = random.randint(min, max)
            fw.write('%d%s%d=' % (a, op, b) + '\n')


generate(['+'], 0, 100000, 1000000, save_path='../data/arithmetic/train.tsv')
generate(['+'], 0, 100000, 2000, save_path='../data/arithmetic/dev.tsv')
# generate_test(['+'],0,100,1000,save_path='../data/arithmetic/test.tsv')
