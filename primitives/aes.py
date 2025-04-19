from primitives.primitives import Permutation, Block_cipher
import operators.operators as op


# The AES internal permutation  
class AES_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the AES internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        if nbr_rounds==None: nbr_rounds=10
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, 8
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        full_rounds = 10
        if nbr_rounds == full_rounds: self.rounds_python_code_if_unrolled, self.rounds_c_code_if_unrolled = {"STATE": [[1, f"if i < {full_rounds-1}:"], [full_rounds, f"elif i == {full_rounds-1}:"]]}, {"STATE": [[1, f"if (i < {full_rounds-1})"+"{"], [full_rounds, f"else if (i == {full_rounds-1})"+"{"]]}
        S = self.states["STATE"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):             
                S.SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                S.PermutationLayer("SR", i, 1, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]) # Shiftrows layer
                if i != full_rounds: S.MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]], "0x1B")  #Mixcolumns layer
                else: S.AddIdentityLayer("ID", i, 2)     # Identity layer 
                S.AddConstantLayer("AC", i, 3, "xor", [True]*16, [[0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]]*nbr_rounds)  # Constant layer            
     

# The AES block cipher
# Test vector for AES_128
# plaintext = [0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34] 
# key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c] 
# ciphertext = [0x39, 0x25, 0x84, 0x1d, 0x2, 0xdc, 0x9, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0xb, 0x32]
# https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf

# Test vector for AES_192 
# plaintext = [0x6B, 0xC1, 0xBE, 0xE2, 0x2E, 0x40, 0x9F, 0x96, 0xE9, 0x3D, 0x7E, 0x11, 0x73, 0x93, 0x17, 0x2A] 
# key = [0x8E, 0x73, 0xB0, 0xF7, 0xDA, 0x0E, 0x64, 0x52, 0xC8, 0x10, 0xF3, 0x2B, 0x80, 0x90, 0x79, 0xE5, 0x62, 0xF8, 0xEA, 0xD2, 0x52, 0x2C, 0x6B, 0x7B] 
# ciphertext = [0xbd, 0x33, 0x4f, 0x1d, 0x6e, 0x45, 0xf2, 0x5f, 0xf7, 0x12, 0xa2, 0x14, 0x57, 0x1f, 0xa5, 0xcc]
# https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core192.pdf

# Test vector for AES_256
# plaintext = [0x6B, 0xC1, 0xBE, 0xE2, 0x2E, 0x40, 0x9F, 0x96, 0xE9, 0x3D, 0x7E, 0x11, 0x73, 0x93, 0x17, 0x2A] 
# key = [0x60, 0x3D, 0xEB, 0x10, 0x15, 0xCA, 0x71, 0xBE, 0x2B, 0x73, 0xAE, 0xF0,  0x85, 0x7D, 0x77, 0x81, 0x1F, 0x35, 0x2C, 0x07, 0x3B, 0x61, 0x08, 0xD7, 0x2D, 0x98, 0x10, 0xA3, 0x09, 0x14, 0xDF, 0xF4]
# ciphertext = [0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c, 0x6, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8]
# https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_Core256.pdf

class AES_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the AES block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize), e.g., (64, 128)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=10 if version[1]==128 else 12 if version[1]==192 else 14 if version[1]==256  else None
        nbr_rounds += 1
        if represent_mode==0: 
            perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
            if k_bitsize==128:
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (7, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
                k_nbr_rounds, k_perm = nbr_rounds, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,13,14,15,12]
                full_rounds=11
            if k_bitsize==192: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (9, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
                k_nbr_rounds, k_perm = int((nbr_rounds+1)/1.5),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,21,22,23,20]
                full_rounds=13
            if k_bitsize==256: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (13, int(16*k_bitsize / p_bitsize), 8, 8),  (1, 16, 0, 8) 
                k_nbr_rounds, k_perm = int((nbr_rounds+1)/2),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,29,30,31,28]
                full_rounds=15
            nk = int(k_bitsize/32)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        S = self.states["STATE"]
        KS = self.states["KEY_STATE"]
        SK = self.states["SUBKEYS"]

        constant_table =self.gen_rounds_constant_table()

        # create constraints
        if represent_mode==0:
            # subkeys extraction
            for i in range(1,nbr_rounds+1): 
                if k_bitsize==128: extracted_bits = KS.vars[i][0]    
                elif k_bitsize==192: extracted_bits = KS.vars[1+int(i/3)*2][0][0:16] if i%3 == 1 else (KS.vars[1+int(i/3)*2][0][16:24] + KS.vars[2+int(i/3)*2][0][0:8]) if i%3 == 2 else KS.vars[2+int((i-1)/3)*2][0][8:24]
                elif k_bitsize==256: extracted_bits = KS.vars[int((i+1)/2)][0] if i%2 != 0 else KS.vars[int((i+1)/2)][0][16:32]
                SK.ExtractionLayer("SK_EX", i, 0, [j for j in range(16)], extracted_bits)
            
            # key schedule    
            for i in range(1,k_nbr_rounds): 
                KS.PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                KS.SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                KS.AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[True]*4, constant_table)  # Constant layer
                KS.SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                KS.SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                KS.SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                KS.SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                if k_bitsize==192: 
                    KS.SingleOperatorLayer("K_XOR", i, 7, op.bitwiseXOR, [[j, j+4] for j in range(12,16)],  [j for j in range(16,20)]) # XOR layer 
                    KS.SingleOperatorLayer("K_XOR", i, 8, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                    if i == k_nbr_rounds-1:
                        for j in range(2*(nbr_rounds % 3)):
                            KS.constraints[i][8-j] = []
                            KS.AddIdentityLayer("ID", i, 8-j)     # Identity layer

                elif k_bitsize==256:
                    KS.PermutationLayer("K_P", i, 7, [i for i in range(36)]+[12,13,14,15]) # Permutation layer
                    KS.SboxLayer("K_SB", i, 8, op.AES_Sbox, mask=([0 for _ in range(36)] + [1, 1, 1, 1])) # Sbox layer   
                    KS.SingleOperatorLayer("K_XOR", i, 9, op.bitwiseXOR, [[j, j+20] for j in range(16, 20)],  [j for j in range(16, 20)]) # XOR layer 
                    KS.SingleOperatorLayer("K_XOR", i, 10, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                    KS.SingleOperatorLayer("K_XOR", i, 11, op.bitwiseXOR, [[j, j+4] for j in range(20,24)],  [j for j in range(24,28)]) # XOR layer 
                    KS.SingleOperatorLayer("K_XOR", i, 12, op.bitwiseXOR, [[j, j+4] for j in range(24,28)],  [j for j in range(28,32)]) # XOR layer 
                    if i == k_nbr_rounds-1 and nbr_rounds % 2 == 1:
                        for j in range(7, 13):
                            KS.constraints[i][j] = []
                            KS.AddIdentityLayer("ID", i, j)     # Identity layer
                            
            # Internal permutation
            S.AddRoundKeyLayer("ARK", 1, 0, op.bitwiseXOR, SK, mask=[1 for i in range(16)])  # AddRoundKey layer   
            S.AddIdentityLayer("ID", 1, 1)     # Identity layer 
            S.AddIdentityLayer("ID", 1, 2)     # Identity layer 
            S.AddIdentityLayer("ID", 1, 3)     # Identity layer 
            for i in range(2,nbr_rounds+1):   
                S.SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                S.PermutationLayer("SR", i, 1, perm_s) # Shiftrows layer
                if i != full_rounds: S.MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]], "0x1B")  #Mixcolumns layer
                else: S.AddIdentityLayer("ID", i, 2) # Identity layer 
                S.AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, SK, mask=[1 for i in range(16)])  # AddRoundKey layer   
                
        # Generate python and c code for round function if unrolled
        self.rounds_code_if_unrolled(k_bitsize, full_rounds)

    def gen_rounds_constant_table(self):
        constant_table = []
        Rcon = [0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000]
        for i in range(1,self.states["KEY_STATE"].nbr_rounds): 
            constant_table.append([Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff])      
        return constant_table
    
    def rounds_code_if_unrolled(self, k_bitsize, full_rounds):
        nbr_rounds = self.states['STATE'].nbr_rounds
        if nbr_rounds >= 2: 
            self.rounds_python_code_if_unrolled["STATE"] = [[1, "if i == 0:"]]
            self.rounds_c_code_if_unrolled["STATE"] = [[1, "if (i == 0)"+"{"]]
            if nbr_rounds < full_rounds: 
                self.rounds_python_code_if_unrolled["STATE"].append([2, f"elif 1 <= i <= {nbr_rounds-1}:"])
                self.rounds_c_code_if_unrolled["STATE"].append([2, f"else if (i <= {nbr_rounds-1})"+"{"])
            elif nbr_rounds == full_rounds:  
                self.rounds_python_code_if_unrolled["STATE"].append([2, f"elif 1 <= i <= {nbr_rounds-2}:"])
                self.rounds_python_code_if_unrolled["STATE"].append([nbr_rounds, f"elif i == {nbr_rounds-1}:"])
                self.rounds_c_code_if_unrolled["STATE"].append([2, f"else if (i <= {nbr_rounds-2})"+"{"])
                self.rounds_c_code_if_unrolled["STATE"].append([nbr_rounds, f"else if (i == {nbr_rounds-1})"+"{"])
        if k_bitsize==128:
            self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, f"if i < {self.states['KEY_STATE'].nbr_rounds-1}:"]]
            self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, f"if (i < {self.states['KEY_STATE'].nbr_rounds-1})"+"{"]]
        elif k_bitsize==256:
            if nbr_rounds >= 2: 
                self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, "if i % 2 == 1:\ni=(i/2)"]]
                self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, "if (i % 2 == 1)"+"{"]]
        

    
