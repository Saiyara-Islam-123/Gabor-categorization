import pandas as pd
import os
import random


def create_non_xab_pairs():
    A = []
    B = []
    types = []

    Bs_in = {}

    for i in range(200):
        index_for_A = i
        index_for_B = random.randint(0, 199)
        while index_for_A == index_for_B or index_for_B in Bs_in:
            index_for_B = random.randint(0, 199)

        [imA, _] = os.listdir(f"..\..\GABORS_400\gabors_1\experimentFiles\gabors\\testing\pair{index_for_A}")
        [_, imB] = os.listdir(f"..\..\GABORS_400\gabors_1\experimentFiles\gabors\\testing\pair{index_for_B}")
        A.append(f"..\..\GABORS_400\gabors_1\experimentFiles\gabors\\testing\pair{index_for_A}"+imA)
        B.append(f"..\..\GABORS_400\gabors_1\experimentFiles\gabors\\testing\pair{index_for_B}"+imB)

        if ("cat_0" in imA and "cat_1" in imB) or ("cat_1" in imA and "cat_0" in imB):
            types.append("between")
        elif "cat_0" in imA and "cat_0" in imB:
            types.append("within_0")
        elif "cat_1" in imA and "cat_1" in imB:
            types.append("within_1")

    df = pd.DataFrame()
    df["A"] = A
    df["B"] = B
    df["type"] = types
    df.to_csv("non_xab_pairs")

if __name__ == "__main__":
    create_non_xab_pairs()
    df = pd.read_csv("non_xab_pairs")
    category_counts = df['type'].value_counts()
    print(category_counts)

    df2 = pd.read_csv("complete_xab_results.csv")
    category_counts2 = df2['type'].value_counts()
    print(category_counts2)