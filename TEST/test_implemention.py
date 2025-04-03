import sys
import os
import subprocess
import ctypes
import numpy as np
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import OCP
import implementations.implementations as imp 


def test_python(cipher, cipher_name, plaintext, key, test_ciphertext):
    print(f"****************TEST PYTHON IMPLEMENTATION of {cipher_name}****************")
    imp.generate_implementation(cipher,"files/" + cipher_name + ".py", "python")
    try:
        imp_cipher = importlib.import_module(f"files.{cipher_name}")
    except ImportError:
        print(f"Implementation module files.{cipher_name} version cannot be loaded.\n")
    func = getattr(imp_cipher, f"{cipher.name}")   
    ciphertext = [0 for _ in range(len(plaintext))]
    func(plaintext, key, ciphertext)
    if ciphertext == test_ciphertext:
        return True
    else:
        print(f'Wrong!\nciphertext = {[hex(i) for i in ciphertext]}\ntest_ciphertext={test_ciphertext}') 
        return False
    

def test_c(cipher, cipher_name, plaintext, key, test_ciphertext):
    print(f"****************TEST C IMPLEMENTATION of {cipher_name}****************")
    imp.generate_implementation(cipher,"files/" + cipher_name + ".c", "c")
    if cipher.inputs["plaintext"][0].bitsize <= 8:
        dtype_np = np.uint8
        dtype_ct = ctypes.c_uint8
    elif cipher.inputs["plaintext"][0].bitsize <= 32:
        dtype_np = np.uint32
        dtype_ct = ctypes.c_uint32
    elif cipher.inputs["plaintext"][0].bitsize <= 64:
        dtype_np = np.uint64
        dtype_ct = ctypes.c_uint64
    else:
        dtype_np = np.uint128
        dtype_ct = ctypes.c_uint128
    plaintext = np.array(plaintext, dtype=dtype_np)
    key = np.array(key, dtype=dtype_np)
    ciphertext = np.zeros(len(plaintext), dtype=dtype_np)
    test_ciphertext = np.array(test_ciphertext, dtype=dtype_np)

    compile_command = f"gcc files/{cipher_name}.c -o files/{cipher_name}.out"
    compile_process = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if compile_process.returncode != 0:
        print(f"[ERROR] Compilation failed for {cipher.name}.c")
        return False
    
    func = getattr(ctypes.CDLL(f"files/{cipher_name}.out"), cipher.name)
    func.argtypes = [ctypes.POINTER(dtype_ct)] * 3
    func(
        plaintext.ctypes.data_as(ctypes.POINTER(dtype_ct)),
        key.ctypes.data_as(ctypes.POINTER(dtype_ct)),
        ciphertext.ctypes.data_as(ctypes.POINTER(dtype_ct))
    )

    if np.array_equal(ciphertext, test_ciphertext):
        return True
    else:
        print(f'Wrong!\nciphertext = {[hex(i) for i in ciphertext]}\ntest_ciphertext={test_ciphertext}') 
        return False
    
def test_speck():
    for version in [[32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]]:
        
        cipher = OCP.TEST_SPECK_BLOCKCIPHER(r=None, version=version)
        
        # test vectors from https://github.com/inmcm/Simon_Speck_Ciphers/blob/master/Python/simonspeckciphers/tests/test_simonspeck.py
        if version == [32, 64]:
            plaintext = [0x6574, 0x694c]
            key = [0x1918, 0x1110, 0x0908, 0x0100]
            ciphertext = [0xa868, 0x42f2]
        elif version == [48, 72]: 
            plaintext = [0x20796c, 0x6c6172]
            key = [0x121110, 0x0a0908, 0x020100]
            ciphertext = [0xc049a5, 0x385adc]
        elif version == [48, 96]: 
            plaintext = [0x6d2073, 0x696874]
            key = [0x1a1918, 0x121110, 0x0a0908, 0x020100]
            ciphertext = [0x735e10, 0xb6445d]
        elif version == [64, 96]: 
            plaintext = [0x74614620, 0x736e6165]
            key = [0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x9f7952ec, 0x4175946c]
        elif version == [64, 128]: 
            plaintext = [0x3b726574, 0x7475432d]
            key = [0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x8c6fa548, 0x454e028b]
        elif version == [96, 96]: 
            plaintext = [0x65776f68202c, 0x656761737520]
            key = [0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x9e4d09ab7178, 0x62bdde8f79aa]
        elif version == [96, 144]: 
            plaintext = [0x656d6974206e, 0x69202c726576]
            key = [0x151413121110, 0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x2bf31072228a, 0x7ae440252ee6]
        elif version == [128, 128]: 
            plaintext = [0x6c61766975716520, 0x7469206564616d20]
            key = [0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0xa65d985179783265, 0x7860fedf5c570d18]
        elif version == [128, 192]: 
            plaintext = [0x7261482066656968, 0x43206f7420746e65]
            key = [0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x1be4cf3a13135566, 0xf9bc185de03c1886]
        elif version == [128, 256]: 
            plaintext = [0x65736f6874206e49, 0x202e72656e6f6f70]
            key = [0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x4109010405c0f53e, 0x4eeeb48d9c188f43]

        # test of python implementation
        print(test_python(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_python(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
    
        # test of C implementation
        print(test_c(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_c(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
        

def test_simon():
    for version in [[32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]]:
        
        cipher = OCP.TEST_SIMON_BLOCKCIPHER(r=None, version=version)

        # test vectors from https://github.com/inmcm/Simon_Speck_Ciphers/blob/master/Python/simonspeckciphers/tests/test_simonspeck.py
        if version == [32, 64]:
            plaintext = [0x6565, 0x6877] 
            key = [0x1918, 0x1110, 0x0908, 0x0100] 
            ciphertext = [0xc69b, 0xe9bb]
        elif version == [48, 72]: 
            plaintext = [0x612067, 0x6e696c]
            key = [0x121110, 0x0a0908, 0x020100]
            ciphertext = [0xdae5ac, 0x292cac]
        elif version == [48, 96]: 
            plaintext = [0x726963, 0x20646e]
            key = [0x1a1918, 0x121110, 0x0a0908, 0x020100]
            ciphertext = [0x6e06a5, 0xacf156]
        elif version == [64, 96]: 
            plaintext = [0x6f722067, 0x6e696c63]
            key = [0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x5ca2e27f, 0x111a8fc8]
        elif version == [64, 128]: 
            plaintext = [0x656b696c, 0x20646e75]
            key = [0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x44c8fc20, 0xb9dfa07a]
        elif version == [96, 96]: 
            plaintext = [0x2072616c6c69, 0x702065687420]
            key = [0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x602807a462b4, 0x69063d8ff082]
        elif version == [96, 144]: 
            plaintext = [0x746168742074, 0x73756420666f]
            key = [0x151413121110, 0x0d0c0b0a0908, 0x050403020100]
            ciphertext =  [0xecad1c6c451e, 0x3f59c5db1ae9]
        elif version == [128, 128]: 
            plaintext = [0x6373656420737265, 0x6c6c657661727420]
            key = [0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x49681b1e1e54fe3f, 0x65aa832af84e0bbc]
        elif version == [128, 192]: 
            plaintext = [0x206572656874206e, 0x6568772065626972]
            key = [0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0xc4ac61effcdc0d4f, 0x6c9c8d6e2597b85b]
        elif version == [128, 256]: 
            plaintext = [0x74206e69206d6f6f, 0x6d69732061207369]
            key = [0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x8d2b5579afc8a3a0, 0x3bf72a87efe7b868]

        # test of python implementation
        print(test_python(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_python(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
    
        # test of C implementation
        print(test_c(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_c(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))


def test_skinny():
    for version in [[64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]]:
        
        cipher = OCP.TEST_SKINNY_BLOCKCIPHER(r=None, version=version)

        # test vectors from https://eprint.iacr.org/2016/660.pdf and https://github.com/inmcm/skinny_cipher/blob/master/Python/skinny.py
        if version == [64, 64]:
            plaintext = [0x0, 0x6, 0x0, 0x3, 0x4, 0xf, 0x9, 0x5, 0x7, 0x7, 0x2, 0x4, 0xd, 0x1, 0x9, 0xd] 
            key = [0xf, 0x5, 0x2, 0x6, 0x9, 0x8, 0x2, 0x6, 0xf, 0xc, 0x6, 0x8, 0x1, 0x2, 0x3, 0x8] 
            ciphertext = [0xb, 0xb, 0x3, 0x9, 0xd, 0xf, 0xb, 0x2, 0x4, 0x2, 0x9, 0xb, 0x8, 0xa, 0xc, 0x7]
        elif version == [128, 128]:
            plaintext = [0xf2, 0x0a, 0xdb, 0x0e, 0xb0, 0x8b, 0x64, 0x8a, 0x3b, 0x2e, 0xee, 0xd1, 0xf0, 0xad, 0xda, 0x14]
            key = [0x4f, 0x55, 0xcf, 0xb0, 0x52, 0x0c, 0xac, 0x52, 0xfd, 0x92, 0xc1, 0x5f, 0x37, 0x07, 0x3e, 0x93] 
            ciphertext = [0x22, 0xff, 0x30, 0xd4, 0x98, 0xea, 0x62, 0xd7, 0xe4, 0x5b, 0x47, 0x6e, 0x33, 0x67, 0x5b, 0x74]
        elif version == [64, 128]:
            plaintext = [0xc, 0xf, 0x1, 0x6, 0xc, 0xf, 0xe, 0x8, 0xf, 0xd, 0x0, 0xf, 0x9, 0x8, 0xa, 0xa] 
            key = [0x9, 0xe, 0xb, 0x9, 0x3, 0x6, 0x4, 0x0, 0xd, 0x0, 0x8, 0x8, 0xd, 0xa, 0x6, 0x3, 0x7, 0x6, 0xa, 0x3, 0x9, 0xd, 0x1, 0xc, 0x8, 0xb, 0xe, 0xa, 0x7, 0x1, 0xe, 0x1] 
            ciphertext = [0x6, 0xc, 0xe, 0xd, 0xa, 0x1, 0xf, 0x4, 0x3, 0xd, 0xe, 0x9, 0x2, 0xb, 0x9, 0xe]
        elif version == [64, 192]:
            plaintext = [0x5, 0x3, 0x0, 0xc, 0x6, 0x1, 0xd, 0x3, 0x5, 0xe, 0x8, 0x6, 0x6, 0x3, 0xc, 0x3]
            key = [0xe, 0xd, 0x0, 0x0, 0xc, 0x8, 0x5, 0xb, 0x1, 0x2, 0x0, 0xd, 0x6, 0x8, 0x6, 0x1, 0x8, 0x7, 0x5, 0x3, 0xe, 0x2, 0x4, 0xb, 0xf, 0xd, 0x9, 0x0, 0x8, 0xf, 0x6, 0x0, 0xb, 0x2, 0xd, 0xb, 0xb, 0x4, 0x1, 0xb, 0x4, 0x2, 0x2, 0xd, 0xf, 0xc, 0xd, 0x0]
            ciphertext = [0xd, 0xd, 0x2, 0xc, 0xf, 0x1, 0xa, 0x8, 0xf, 0x3, 0x3, 0x0, 0x3, 0x0, 0x3, 0xc]
        elif version == [128, 256]:
            plaintext = [0x3a, 0x0c, 0x47, 0x76, 0x7a, 0x26, 0xa6, 0x8d, 0xd3, 0x82, 0xa6, 0x95, 0xe7, 0x02, 0x2e, 0x25]    
            key = [0x00, 0x9c, 0xec, 0x81, 0x60, 0x5d, 0x4a, 0xc1, 0xd2, 0xae, 0x9e, 0x30, 0x85, 0xd7, 0xa1, 0xf3, 0x1a, 0xc1, 0x23, 0xeb, 0xfc, 0x00, 0xfd, 0xdc, 0xf0, 0x10, 0x46, 0xce, 0xed, 0xdf, 0xca, 0xb3] 
            ciphertext = [0xb7, 0x31, 0xd9, 0x8a, 0x4b, 0xde, 0x14, 0x7a, 0x7e, 0xd4, 0xa6, 0xf1, 0x6b, 0x9b, 0x58, 0x7f]
        elif version == [128, 384]:
            plaintext = [0xa3,0x99,0x4b,0x66,0xad,0x85,0xa3,0x45,0x9f,0x44,0xe9,0x2b,0x08,0xf5,0x50,0xcb]
            key = [0xdf,0x88,0x95,0x48,0xcf,0xc7,0xea,0x52,0xd2,0x96,0x33,0x93,0x01,0x79,0x74,0x49, 0xab,0x58,0x8a,0x34,0xa4,0x7f,0x1a,0xb2,0xdf,0xe9,0xc8,0x29,0x3f,0xbe,0xa9,0xa5, 0xab,0x1a,0xfa,0xc2,0x61,0x10,0x12,0xcd,0x8c,0xef,0x95,0x26,0x18,0xc3,0xeb,0xe8]
            ciphertext = [0x94, 0xec, 0xf5, 0x89, 0xe2, 0x1, 0x7c, 0x60, 0x1b, 0x38, 0xc6, 0x34, 0x6a, 0x10, 0xdc, 0xfa]       

        # test of python implementation
        print(test_python(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_python(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
    
        # test of C implementation
        print(test_c(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_c(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))


def test_aes():
    for version in [[128, 128], [128, 192], [128, 256]]:
        
        cipher = OCP.TEST_AES_BLOCKCIPHER(r=None, version=version)

        # test vectors from # https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf,  https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core192.pdf, https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core256.pdf
        if version == [128, 128]:
            plaintext = [0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34] 
            key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c] 
            ciphertext = [0x39, 0x25, 0x84, 0x1d, 0x2, 0xdc, 0x9, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0xb, 0x32]
        elif version == [128, 192]:
            plaintext = [0x6B, 0xC1, 0xBE, 0xE2, 0x2E, 0x40, 0x9F, 0x96, 0xE9, 0x3D, 0x7E, 0x11, 0x73, 0x93, 0x17, 0x2A] 
            key = [0x8E, 0x73, 0xB0, 0xF7, 0xDA, 0x0E, 0x64, 0x52, 0xC8, 0x10, 0xF3, 0x2B, 0x80, 0x90, 0x79, 0xE5, 0x62, 0xF8, 0xEA, 0xD2, 0x52, 0x2C, 0x6B, 0x7B] 
            ciphertext = [0xbd, 0x33, 0x4f, 0x1d, 0x6e, 0x45, 0xf2, 0x5f, 0xf7, 0x12, 0xa2, 0x14, 0x57, 0x1f, 0xa5, 0xcc]
        elif version == [128, 256]:
            plaintext = [0x6B, 0xC1, 0xBE, 0xE2, 0x2E, 0x40, 0x9F, 0x96, 0xE9, 0x3D, 0x7E, 0x11, 0x73, 0x93, 0x17, 0x2A] 
            key = [0x60, 0x3D, 0xEB, 0x10, 0x15, 0xCA, 0x71, 0xBE, 0x2B, 0x73, 0xAE, 0xF0,  0x85, 0x7D, 0x77, 0x81, 0x1F, 0x35, 0x2C, 0x07, 0x3B, 0x61, 0x08, 0xD7, 0x2D, 0x98, 0x10, 0xA3, 0x09, 0x14, 0xDF, 0xF4]
            ciphertext = [0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c, 0x6, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8]

        # test of python implementation
        print(test_python(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_python(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
    
        # test of C implementation
        print(test_c(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_c(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))


def test_gift():
    for version in [[64, 128], [128, 128]]:
        
        cipher = OCP.TEST_GIFT_BLOCKCIPHER(r=None, version=version)

        # test vectors from # https://github.com/giftcipher/gift/tree/master/implementations/test%20vectors
        if version == [64, 128]:
            plaintext =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            key =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ciphertext =  [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
            # plaintext =  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            # key =  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            # ciphertext =  [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            # plaintext =  [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]
            # key =  [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            # ciphertext =  [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
        elif version == [128, 128]:
            plaintext = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ciphertext = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
            # plaintext =  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            # key =  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            # ciphertext = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
            # plaintext =  [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
            # key =  [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
            # ciphertext = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
        
        # test of python implementation
        print(test_python(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_python(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))
    
        # test of C implementation
        print(test_c(cipher, cipher.name, plaintext, key, ciphertext))
        print(test_c(cipher, cipher.name + "_unrolled", plaintext, key, ciphertext))


if __name__ == '__main__':
    test_speck()
    test_simon()
    test_skinny()
    test_aes()
    test_gift()
    
    



