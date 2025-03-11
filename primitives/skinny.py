from primitives.primitives import Permutation, Block_cipher
import operators.operators as op

# The Skinny internal permutation       
class Skinny_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None):
        """
        Initialize the Skinny internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation (e.g., 64 or 128)
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds (optional)
        """
        
        # define the parameters
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==64 else 64 if version==128 else None
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, int(p_bitsize/16)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        round_constants = [[1, 0, 2], [3, 0, 2], [7, 0, 2], [15, 0, 2], [15, 1, 2], [14, 3, 2], [13, 3, 2], [11, 3, 2], [7, 3, 2], [15, 2, 2], [14, 1, 2], [12, 3, 2], [9, 3, 2], [3, 3, 2], [7, 2, 2], [14, 0, 2], [13, 1, 2], [10, 3, 2], [5, 3, 2], [11, 2, 2], [6, 1, 2], [12, 2, 2], [8, 1, 2], [0, 3, 2], [1, 2, 2], [2, 0, 2], [5, 0, 2], [11, 0, 2], [7, 1, 2], [14, 2, 2], [12, 1, 2], [8, 3, 2]]
        sbox = op.Skinny_4bit_Sbox if word_bitsize==4 else op.Skinny_8bit_Sbox 

        # create constraints
        for i in range(1,nbr_rounds+1):              
            self.states["STATE"].SboxLayer("SB", i, 0, sbox)           
            self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer            
            self.states["STATE"].PermutationLayer("SR", i, 2, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
            self.states["STATE"].MatrixLayer("MC", i, 3, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer



# The Skinny block cipher        
class Skinny_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None):
        """
        Initializes the Skinny block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize), e.g., (64, 128)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param model_type: Model type
        """
        
        # define the parameters
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(64,64) else 36 if (version[0],version[1])==(64,128) else 40 if (version[0],version[1])==(64,192)  else 40 if (version[0],version[1])==(128,128)  else 48 if (version[0],version[1])==(128,256)  else 56 if (version[0],version[1])==(128,384) else None
        self.tweak_size = int(k_bitsize/p_bitsize)
        if self.tweak_size ==1: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (1, int(16*k_bitsize / p_bitsize), 0, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
        elif self.tweak_size == 2: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (3, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
        elif self.tweak_size ==3: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (5, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
        k_nbr_rounds = nbr_rounds if self.tweak_size == 1 else nbr_rounds + 1
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        round_constants = [[1, 0, 2], [3, 0, 2], [7, 0, 2], [15, 0, 2], [15, 1, 2], [14, 3, 2], [13, 3, 2], [11, 3, 2], [7, 3, 2], [15, 2, 2], [14, 1, 2], [12, 3, 2], [9, 3, 2], [3, 3, 2], [7, 2, 2], [14, 0, 2], [13, 1, 2], [10, 3, 2], [5, 3, 2], [11, 2, 2], [6, 1, 2], [12, 2, 2], [8, 1, 2], [0, 3, 2], [1, 2, 2], [2, 0, 2], [5, 0, 2], [11, 0, 2], [7, 1, 2], [14, 2, 2], [12, 1, 2], [8, 3, 2], [1, 3, 2], [3, 2, 2], [6, 0, 2], [13, 0, 2], [11, 1, 2], [6, 3, 2], [13, 2, 2], [10, 1, 2], [4, 3, 2], [9, 2, 2], [2, 1, 2], [4, 2, 2], [8, 0, 2], [1, 1, 2], [2, 2, 2], [4, 0, 2], [9, 0, 2], [3, 1, 2], [6, 2, 2], [12, 0, 2], [9, 1, 2], [2, 3, 2], [5, 2, 2], [10, 0, 2]]
        k_perm_T = [i + 16 * j for j in range(self.tweak_size) for i in [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]]    
        if s_word_bitsize == 4:
            mat1 = [[1,None],[2,None],[3,None],[0,1]]
            mat2 = [0,3],[0,None],[1,None],[2,None]
            sbox = op.Skinny_4bit_Sbox
        elif s_word_bitsize == 8:
            mat1 = [[1,None],[2,None],[3,None],[4,None],[5,None],[6,None],[7,None],[0,2]]
            mat2 = [[1,7],[0,None],[1,None],[2,None],[3,None],[4,None],[5,None],[6,None]]
            sbox = op.Skinny_8bit_Sbox
        if self.tweak_size >= 2: self.states_implementation_order = ["KEY_STATE", "SUBKEYS", "STATE"] 
        
        # create constraints

        # Subkey extraction
        for i in range(1,nbr_rounds+1): 
            if self.tweak_size == 1:
                self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(8)], self.states["KEY_STATE"].vars[i][0])
            elif self.tweak_size >= 2:
                self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(self.tweak_size*16,self.tweak_size*16+8)], self.states["KEY_STATE"].vars[i+1][0])
            
        # Key schedule
        if self.tweak_size == 1:
            for i in range(1, k_nbr_rounds): 
                self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
        elif self.tweak_size == 2: 
            for i in range(1, k_nbr_rounds): 
                if i == 1: 
                    self.states["KEY_STATE"].AddIdentityLayer("ID", i, 0)     # Identity layer 
                    self.states["KEY_STATE"].AddIdentityLayer("ID", i, 1)     # Identity layer 
                else:
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 2, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(32,40)]) # XOR layer 
        elif self.tweak_size == 3:
            for i in range(1, k_nbr_rounds): 
                if i == 1: 
                    self.states["KEY_STATE"].AddIdentityLayer("ID", i, 0)     # Identity layer 
                    self.states["KEY_STATE"].AddIdentityLayer("ID", i, 1)     # Identity layer
                    self.states["KEY_STATE"].AddIdentityLayer("ID", i, 2)     # Identity layer 
                else:
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR2", i, 2, op.bitwiseXOR, [[j,j] for j in range(32,40)], [j for j in range(32,40)], mat=mat2)    
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(48,56)]) # XOR layer 
                self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j,16+j] for j in range(32,40)], [j for j in range(48,56)]) # XOR layer 
                
        # Internal permutation
        for i in range(1,nbr_rounds+1):  
            self.states["STATE"].SboxLayer("SB", i, 0, sbox) # Sbox layer                        
            self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer            
            self.states["STATE"].AddRoundKeyLayer("ARK", i, 2, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(8)])  # AddRoundKey layer   
            self.states["STATE"].PermutationLayer("SR", i, 3, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
            self.states["STATE"].MatrixLayer("MC", i, 4, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer

        # Generate python and c code for round function if unrolled
        self.rounds_code_if_unrolled()


    def rounds_code_if_unrolled(self):
        if (self.tweak_size == 2 or self.tweak_size == 3) and self.states["KEY_STATE"].nbr_rounds >= 2: 
            self.rounds_python_code_if_unrolled = {"KEY_STATE": [[1, "if i == 0:"], [2, "else:"]]}
            self.rounds_c_code_if_unrolled = {"KEY_STATE": [[1, "if (i == 0)"+"{"], [2, "else{"]]}
            