from primitives.primitives import Permutation
import operators.operators as op


# The ASCON internal permutation             
class ASCON_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the ASCON internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        if nbr_rounds==None: nbr_rounds=12
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 320, 320, 1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        constant_table = self.gen_rounds_constant_table()
        
        # create constraints
        if represent_mode==0: 
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].AddConstantLayer("C", i, 0, "xor", [None]*184+[True]*8, constant_table)  # Constant layer      
                self.states["STATE"].SboxLayer("SB", i, 1, op.ASCON_Sbox, index=[[k+j*64 for j in range(5)] for k in range(64)])  # Sbox layer            
                self.states["STATE"].SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[(45+j)%64, (36+j)%64] for j in range(64)]+[[(3+j)%64+64, (25+j)%64+64] for j in range(64)]+[[(63+j)%64+128, (58+j)%64+128] for j in range(64)]+[[(54+j)%64+192, (47+j)%64+192] for j in range(64)]+[[(57+j)%64+256, (23+j)%64+256] for j in range(64)], [j for j in range(320,640)]) # XOR layer 
                self.states["STATE"].SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[j, j+320] for j in range(320)], [j for j in range(320)]) # XOR layer 
    
    def gen_rounds_constant_table(self):
        constant_table = []
        cons = [0xf0 - r*0x10 + r*0x1 for r in range(12)]
        for i in range(1,self.states["STATE"].nbr_rounds+1):  
            constant_table.append([((cons[12-self.states["STATE"].nbr_rounds+i-1] >> (7-l)) & 1) for l in range(8)])
        return constant_table