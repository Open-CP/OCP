"""
The aims is given given a fixed nxn matrix "mc" and an arbitrary nxn matrix "state". 
Give me a fast way to compute the matrix multiplication.
"""
from typing import List
def gmul(a,b):
    p = 0
    for c in range(8):
        if b & 1:
            p ^= a
        carry = a & 0x80
        a <<= 1
        if carry:
            a ^= 0x11b
        b >>= 1
    return p

def transpose(arr, n):
    assert len(arr)==n*n
    for r in range(n):
        for c in range(r+1,n):
            arr[r*n + c], arr[c*n+r] = arr[c*n+r],arr[r*n+c]

def generate_ttable(mc:List[int],sbox:List[int], rows:int):
    assert len(mc)==rows**2 
    assert len(sbox)==256 
    mccpy = mc.copy()
    mc = mccpy 
    transpose(mc, rows)
    tables = [[0]*len(sbox) for _ in range(rows)]#each value has self.n bytes
    for t in range(rows):
        for num in range(256):
            tmp = [] 
            for c in range(rows):#take the row of the mc
                con = mc[t*rows + c]
                tmp.append(gmul(sbox[num],con))
            tables[t][num] = int.from_bytes(bytes(tmp), "big")
    return tables 