from primitives.primitives import Permutation, Block_cipher
import operators.operators as op


# The Speck internal permutation  
class Speck_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Speck internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        # define the parameters                
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=22 if version==32 else 22 if version==48 else 26 if version==64 else 28 if version==96 else 32 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 2, 0, p_bitsize>>1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.states["STATE"]
        rotr, rotl = (7, 2) if version == 32 else (8, 3)
        
        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT1", i, 0, ['r', rotr, 0]) # Rotation layer
                S.SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer   
                S.RotationLayer("ROT2", i, 2, ['l', rotl, 1]) # Rotation layer 
                S.SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[0,1]], [1]) # XOR layer 
  

# The Speck block cipher        
class Speck_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Speck block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        # define the parameters                
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=22 if (version[0],version[1])==(32,64) else 22 if (version[0],version[1])==(48,72) else 23 if (version[0],version[1])==(48,96)  else 26 if (version[0],version[1])==(64,96)  else 27 if (version[0],version[1])==(64,128)  else 28 if (version[0],version[1])==(96,96) else 29 if (version[0],version[1])==(96,144) else 32 if (version[0],version[1])==(128,128) else 33 if (version[0],version[1])==(128,192) else 34 if (version[0],version[1])==(128,256) else None
        if represent_mode==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 2, 0, p_bitsize>>1),  (6, int(2*k_bitsize / p_bitsize), 0, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        round_constants = self.gen_rounds_constant_table()
        rotr, rotl = (7, 2) if version == 32 else (8, 3)
        if k_bitsize==p_bitsize: perm, left_k_index, right_k_index = [0,1], 0, 1
        elif k_bitsize==1.5*p_bitsize: perm, left_k_index, right_k_index = [1,0,2], 1, 2
        elif k_bitsize==2*p_bitsize: perm, left_k_index, right_k_index = [2,0,1,3], 2, 3

        S = self.states["STATE"]
        KS = self.states["KEY_STATE"]
        SK = self.states["SUBKEYS"] 

        # create constraints
        if represent_mode==0:         

            for i in range(1,nbr_rounds+1):
                # subkeys extraction
                SK.ExtractionLayer("SK_EX", i, 0, [right_k_index], KS.vars[i][0])
  
                # key schedule
                KS.RotationLayer("ROT1", i, 0, ['r', rotr, left_k_index]) # Rotation layer
                KS.SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[left_k_index, right_k_index]], [left_k_index]) # Modular addition layer   
                KS.RotationLayer("ROT2", i, 2, ['l', rotl, right_k_index]) # Rotation layer 
                KS.AddConstantLayer("C", i, 3, "xor", [True if e==left_k_index else None for e in range(KS.nbr_words)], round_constants)  # Constant layer
                KS.SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[left_k_index, right_k_index]], [right_k_index]) # XOR layer 
                KS.PermutationLayer("SHIFT", i, 5, perm) # key schedule word shift
            
                # Internal permutation
                S.RotationLayer("ROT1", i, 0, ['r', rotr, 0]) # Rotation layer
                S.SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer  
                S.RotationLayer("ROT2", i, 2, ['l', rotl, 1]) # Rotation layer 
                S.AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, SK, [1,0]) # Addroundkey layer 
                S.SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[0,1]], [1]) # XOR layer
         
    def gen_rounds_constant_table(self):
        constant_table = []
        for i in range(1,self.states["STATE"].nbr_rounds+1):    
            constant_table.append([i-1])
        return constant_table
