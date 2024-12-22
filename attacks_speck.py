import primitives as prim
import variables as var
import attacks as at
import time
import pandas as pd
import tool


# ********************* TEST OF SPECK ********************* #   
def TEST_SPECK32_PERMUTATION(r):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation("SPECK32_PERM", 32, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SPECK32_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="k"+str(i)) for i in range(4)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher("SPECK32", [32, 64], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher



def single_key_diff_MILP():
    data = {'Rounds': [], 'Result':[], 'Time(s)': []}
    for r in range(1, 9):
        strat_time = time.time()
        cipher = TEST_SPECK32_PERMUTATION(r)
        # cipher = TEST_SPECK32_BLOCKCIPHER(r)
        obj = at.singlekey_differential_path_search_milp(cipher, r)
        end_time = time.time()
        data['Rounds'].append(r)
        data['Result'].append(int(obj))
        data['Time(s)'].append("{:.2f}".format(end_time - strat_time))
        df = pd.DataFrame(data)
        latex_code = df.to_latex(index=False, header=True, caption=f'Experimental results for {cipher.name} by solving MILP models')
        print(latex_code)


def single_key_diff_SAT():
    data = {'Rounds': [], 'Result':[], 'Time(s)': []}
    for r in range(1, 9):
        strat_time = time.time()
        cipher = TEST_SPECK32_PERMUTATION(r)
        # cipher = TEST_SPECK32_BLOCKCIPHER(r)
        obj = at.singlekey_differential_path_search_sat(cipher, r)
        end_time = time.time()
        data['Rounds'].append(r)
        data['Result'].append(int(obj))
        data['Time(s)'].append("{:.2f}".format(end_time - strat_time))
        df = pd.DataFrame(data)
        latex_code = df.to_latex(index=False, header=True, caption=f'Experimental results for {cipher.name} by solving SAT models')
        print(latex_code)


# def related_key_diff_milp(): # TO DO


# def related_key_diff_sat(): # TO DO



if __name__ == '__main__':
    # single_key_diff_MILP()
    single_key_diff_SAT()