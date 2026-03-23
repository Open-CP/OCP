"""
The aims is given given a fixed nxn matrix "mc" and an arbitrary nxn matrix "state". 
Give me a fast way to compute the matrix multiplication.
"""
from typing import List
def GMUL(a, b, p, d):
	result = 0
	while b > 0:
		if b & 1:
			result ^= a
		a <<= 1
		if a & (1 << d):
			a ^= p
		b >>= 1
	return result & ((1 << d) - 1)

def permute(arr, prr):
    return [arr[p] for p in prr]
def flatten(arr):
    rtn = [] 
    for r in arr: rtn.extend(r)
    return rtn
def transpose(arr, n):
    assert len(arr)==n*n
    for r in range(n):
        for c in range(r+1,n):
            arr[r*n + c], arr[c*n+r] = arr[c*n+r],arr[r*n+c]

def generate_ttable(mc:List[int],sbox:List[int], rows:int, word_size:int, poly:int):
    assert len(mc)==rows**2 
    #assert len(sbox)==256 
    sbox_len = len(sbox)
    mccpy = mc.copy()
    mc = mccpy 
    transpose(mc, rows)
    tables = [[0]*sbox_len for _ in range(rows)]#each value has self.n bytes
    for t in range(rows):
        for num in range(sbox_len):
            tmp = [] 
            for c in range(rows):#take the row of the mc
                con = mc[t*rows + c]
                if con==1:
                    tmp.append(sbox[num])
                else:
                    tmp.append(GMUL(sbox[num],con,poly,word_size))
            tables[t][num] = int.from_bytes(bytes(tmp), "big")
    return tables 