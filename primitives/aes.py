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

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):             
                self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                self.states["STATE"].PermutationLayer("SR", i, 1, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]) # Shiftrows layer
                if i != full_rounds: self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]], "0x1B")  #Mixcolumns layer
                else: self.states["STATE"].AddIdentityLayer("ID", i, 2)     # Identity layer 
                self.states["STATE"].AddConstantLayer("AC", i, 3, "xor", [True]*16, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14])  # Constant layer            
     

# The AES block cipher
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
        full_rounds=11 if version[1]==128 else 13 if version[1]==192 else 15 if version[1]==256  else None
        if represent_mode==0: 
            if k_bitsize==128:
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (7, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==192: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (9, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==256: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (13, int(16*k_bitsize / p_bitsize), 8, 8),  (1, 16, 0, 8) 
        self.p_bitsize, self.k_bitsize = version[0], version[1]
        
        perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        if k_bitsize==128: k_nbr_rounds, k_perm = nbr_rounds, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,13,14,15,12]
        elif k_bitsize==192: k_nbr_rounds, k_perm = int((nbr_rounds+1)/1.5),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,21,22,23,20]
        elif k_bitsize==256: k_nbr_rounds, k_perm = int((nbr_rounds+1)/2),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,29,30,31,28]
        nk = int(k_bitsize/32)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        constant_table =self.gen_rounds_constant_table(k_bitsize)

        # create constraints
        if represent_mode==0:
            # subkeys extraction
            for i in range(1,nbr_rounds+1): 
                if k_bitsize==128: extracted_bits = self.states["KEY_STATE"].vars[i][0]    
                elif k_bitsize==192: extracted_bits = self.states["KEY_STATE"].vars[1+int(i/3)*2][0][0:16] if i%3 == 1 else (self.states["KEY_STATE"].vars[1+int(i/3)*2][0][16:24] + self.states["KEY_STATE"].vars[2+int(i/3)*2][0][0:8]) if i%3 == 2 else self.states["KEY_STATE"].vars[2+int((i-1)/3)*2][0][8:24]
                elif k_bitsize==256: extracted_bits = self.states["KEY_STATE"].vars[int((i+1)/2)][0] if i%2 != 0 else self.states["KEY_STATE"].vars[int((i+1)/2)][0][16:32]
                self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [j for j in range(16)], extracted_bits)
            
            # key schedule    
            for i in range(1,k_nbr_rounds): 
                self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[True]*4, constant_table)  # Constant layer
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                if k_bitsize==192: 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 7, op.bitwiseXOR, [[j, j+4] for j in range(12,16)],  [j for j in range(16,20)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 8, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                elif k_bitsize==256:
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 7, [i for i in range(36)]+[12,13,14,15]) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 8, op.AES_Sbox, mask=([0 for i in range(36)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 9, op.bitwiseXOR, [[j, j+20] for j in range(16, 20)],  [j for j in range(16, 20)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 10, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 11, op.bitwiseXOR, [[j, j+4] for j in range(20,24)],  [j for j in range(24,28)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 12, op.bitwiseXOR, [[j, j+4] for j in range(24,28)],  [j for j in range(28,32)]) # XOR layer 
                   
            # Internal permutation
            self.states["STATE"].AddRoundKeyLayer("ARK", 1, 0, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
            self.states["STATE"].AddIdentityLayer("ID", 1, 1)     # Identity layer 
            self.states["STATE"].AddIdentityLayer("ID", 1, 2)     # Identity layer 
            self.states["STATE"].AddIdentityLayer("ID", 1, 3)     # Identity layer 
            for i in range(2,nbr_rounds+1):   
                self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                self.states["STATE"].PermutationLayer("SR", i, 1, perm_s) # Shiftrows layer
                if i != full_rounds: self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]], "0x1B")  #Mixcolumns layer
                else: self.states["STATE"].AddIdentityLayer("ID", i, 2) # Identity layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
                
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
                self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, "if i % 2 == 1:"]]
                self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, "if (i % 2 == 1)"+"{"]]
        

    
