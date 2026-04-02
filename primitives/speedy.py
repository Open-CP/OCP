from primitives.primitives import Permutation
from operators.Sbox import SPEEDY_Sbox
from operators.boolean_operators import N_XOR
import variables.variables as var


# The SPEEDY internal permutation  
class SPEEDY_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the AES internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        if nbr_rounds==None and version == 192: nbr_rounds=7
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (6, version, 0, 1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])

        l = int(version/6)
        
        S = self.functions["PERMUTATION"]
        constant_table =self.gen_rounds_constant_table()
        if version == 192: alpha = [1, 5, 9, 15, 21, 26]


        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):             
                S.SboxLayer("SB", i, 0, SPEEDY_Sbox,index=[list(range(i, i + 6)) for i in range(0, nbr_words, 6)]) # Sbox layer   
                S.PermutationLayer("SC", i, 1, [0, 7, 14, 21, 28, 35, 6, 13, 20, 27, 34, 41, 12, 19, 26, 33, 40, 47, 18, 25, 32, 39, 46, 53, 24, 31, 38, 45, 52, 59, 30, 37, 44, 51, 58, 65, 36, 43, 50, 57, 64, 71, 42, 49, 56, 63, 70, 77, 48, 55, 62, 69, 76, 83, 54, 61, 68, 75, 82, 89, 60, 67, 74, 81, 88, 95, 66, 73, 80, 87, 94, 101, 72, 79, 86, 93, 100, 107, 78, 85, 92, 99, 106, 113, 84, 91, 98, 105, 112, 119, 90, 97, 104, 111, 118, 125, 96, 103, 110, 117, 124, 131, 102, 109, 116, 123, 130, 137, 108, 115, 122, 129, 136, 143, 114, 121, 128, 135, 142, 149, 120, 127, 134, 141, 148, 155, 126, 133, 140, 147, 154, 161, 132, 139, 146, 153, 160, 167, 138, 145, 152, 159, 166, 173, 144, 151, 158, 165, 172, 179, 150, 157, 164, 171, 178, 185, 156, 163, 170, 177, 184, 191, 162, 169, 176, 183, 190, 5, 168, 175, 182, 189, 4, 11, 174, 181, 188, 3, 10, 17, 180, 187, 2, 9, 16, 23, 186, 1, 8, 15, 22, 29])
                S.SboxLayer("SB", i, 2, SPEEDY_Sbox,index=[list(range(i, i + 6)) for i in range(0, nbr_words, 6)]) # Sbox layer   
                if i != 7: 
                    S.PermutationLayer("SC", i, 3, [0, 7, 14, 21, 28, 35, 6, 13, 20, 27, 34, 41, 12, 19, 26, 33, 40, 47, 18, 25, 32, 39, 46, 53, 24, 31, 38, 45, 52, 59, 30, 37, 44, 51, 58, 65, 36, 43, 50, 57, 64, 71, 42, 49, 56, 63, 70, 77, 48, 55, 62, 69, 76, 83, 54, 61, 68, 75, 82, 89, 60, 67, 74, 81, 88, 95, 66, 73, 80, 87, 94, 101, 72, 79, 86, 93, 100, 107, 78, 85, 92, 99, 106, 113, 84, 91, 98, 105, 112, 119, 90, 97, 104, 111, 118, 125, 96, 103, 110, 117, 124, 131, 102, 109, 116, 123, 130, 137, 108, 115, 122, 129, 136, 143, 114, 121, 128, 135, 142, 149, 120, 127, 134, 141, 148, 155, 126, 133, 140, 147, 154, 161, 132, 139, 146, 153, 160, 167, 138, 145, 152, 159, 166, 173, 144, 151, 158, 165, 172, 179, 150, 157, 164, 171, 178, 185, 156, 163, 170, 177, 184, 191, 162, 169, 176, 183, 190, 5, 168, 175, 182, 189, 4, 11, 174, 181, 188, 3, 10, 17, 180, 187, 2, 9, 16, 23, 186, 1, 8, 15, 22, 29])
                    S.SingleOperatorLayer("MC", i, 4, N_XOR, [[i*6+j, (i+alpha[0])%l*6+j, (i+alpha[1])%l*6+j, (i+alpha[2])%l*6+j, (i+alpha[3])%l*6+j, (i+alpha[4])%l*6+j, (i+alpha[5])%l*6+j] for i in range(l) for j in range(6)], [j for j in range(320)]) # 3-XOR layer 
                    S.AddConstantLayer("AC", i, 5, "xor", [True]*version, constant_table)  # Constant layer            
                else: 
                    S.AddIdentityLayer("ID", i, 3)     # Identity layer
                    S.AddIdentityLayer("ID", i, 4)     # Identity layer
                    S.AddIdentityLayer("ID", i, 5)     # Identity layer
                
        self.test_vectors = self.gen_test_vectors(version)
                
    def gen_rounds_constant_table(self):
        constants = [0x243f6a8885a308d3,0x13198a2e03707344,0xa4093822299f31d0,0x082efa98ec4e6c89,0x452821e638d01377,0xbe5466cf34e90c6c,
                                0xc0ac29b7c97c50dd,0x3f84d5b5b5470917,0x9216d5d98979fb1b,0xd1310ba698dfb5ac,0x2ffd72dbd01adfb7,0xb8e1afed6a267e96,
                                0xba7c9045f12c7f99,0x24a19947b3916cf7,0x0801f2e2858efc16,0x636920d871574e69,0xa458fea3f4933d7e,0x0d95748f728eb658,
                                0x718bcd5882154aee,0x7b54a41dc25a59b5,0x9c30d5392af26013,0xc5d1b023286085f0,0xca417918b8db38ef,0x8e79dcb0603a180e,
                                0x6c9e0e8bb01e8a3e,0xd71577c1bd314b27,0x78af2fda55605c60,0xe65525f3aa55ab94,0x5748986263e81440,0x55ca396a2aab10b6,
                                0xb4cc5c341141e8ce,0xa15486af7c72e993,0xb3ee1411636fbc2a,0x2ba9c55d741831f6,0xce5c3e169b87931e,0xafd6ba336c24cf5c,
                                0x7a32538128958677,0x3b8f48986b4bb9af,0xc4bfe81b66282193,0x61d809ccfb21a991,0x487cac605dec8032,0xef845d5de98575b1,
                                0xdc262302eb651b88,0x23893e81d396acc5,0x0f6d6ff383f44239,0x2e0b4482a4842004,0x69c8f04a9e1f9b5e,0x21c66842f6e96c9a,
                                0x670c9c61abd388f0,0x6a51a0d2d8542f68,0x960fa728ab5133a3,0x6eef0b6c137a3be4,0xba3bf0507efb2a98,0xa1f1651d39af0176,
                                0x66ca593e82430e88,0x8cee8619456f9fb4,0x7d84a5c33b8b5ebe,0xe06f75d885c12073,0x401a449f56c16aa6,0x4ed3aa62363f7706,
                                0x1bfedf72429b023d,0x37d0d724d00a1248,0xdb0fead349f1c09b,0x075372c980991b7b,0x25d479d8f6e8def7,0xe3fe501ab6794c3b,
                                0x976ce0bd04c006ba,0xc1a94fb6409f60c4,0x5e5c9ec2196a2463,0x68fb6faf3e6c53b5,0x1339b2eb3b52ec6f,0x6dfc511f9b30952c,
                                0xcc814544af5ebd09,0xbee3d004de334afd,0x660f2807192e4bb3,0xc0cba85745c8740f,0xd20b5f39b9d3fbdb,0x5579c0bd1a60320a,
                                0xd6a100c6402c7279,0x679f25fefb1fa3cc,0x8ea5e9f8db3222f8,0x3c7516dffd616b15,0x2f501ec8ad0552ab,0x323db5fafd238760,
                                0x53317b483e00df82,0x9e5c57bbca6f8ca0,0x1a87562edf1769db,0xd542a8f6287effc3,0xac6732c68c4f5573,0x695b27b0bbca58c8,
                                0xe1ffa35db8f011a0,0x10fa3d98fd2183b8,0x4afcb56c2dd1d35b,0x9a53e479b6f84565,0xd28e49bc4bfb9790,0xe1ddf2daa4cb7e33,
                                0x62fb1341cee4c6e8,0xef20cada36774c01,0xd07e9efe2bf11fb4,0x95dbda4dae909198,0xeaad8e716b93d5a0,0xd08ed1d0afc725e0,
                                0x8e3c5b2f8e7594b7,0x8ff6e2fbf2122b64,0x8888b812900df01c,0x4fad5ea0688fc31c,0xd1cff191b3a8c1ad,0x2f2f2218be0e1777,
                                0xea752dfe8b021fa1,0xe5a0cc0fb56f74e8,0x18acf3d6ce89e299,0xb4a84fe0fd13e0b7,0x7cc43b81d2ada8d9,0x165fa26680957705,
                                0x93cc7314211a1477,0xe6ad206577b5fa86,0xc75442f5fb9d35cf,0xebcdaf0c7b3e89a0,0xd6411bd3ae1e7e49,0x00250e2d2071b35e,
                                0x226800bb57b8e0af,0x2464369bf009b91e,0x5563911d59dfa6aa,0x78c14389d95a537f,0x207d5ba202e5b9c5,0x832603766295cfa9,
                                0x11c819684e734a41,0xb3472dca7b14a94a,0x1b5100529a532915,0xd60f573fbc9bc6e4,0x2b60a47681e67400,0x08ba6fb5571be91f,
                                0xf296ec6b2a0dd915,0xb6636521e7b9f9b6,0xff34052ec5855664,0x53b02d5da99f8fa1,0x08ba47996e85076a,0x4b7a70e9b5b32944,
                                0xdb75092ec4192623,0xad6ea6b049a7df7d,0x9cee60b88fedb266,0xecaa8c71699a17ff,0x5664526cc2b19ee1,0x193602a575094c29,
                                0xa0591340e4183a3e,0x3f54989a5b429d65,0x6b8fe4d699f73fd6,0xa1d29c07efe830f5,0x4d2d38e6f0255dc1,0x4cdd20868470eb26,
                                0x6382e9c6021ecc5e,0x09686b3f3ebaefc9,0x3c9718146b6a70a1,0x687f358452a0e286,0xb79c5305aa500737,0x3e07841c7fdeae5c,
                                0x8e7d44ec5716f2b8,0xb03ada37f0500c0d,0xf01c1f040200b3ff,0xae0cf51a3cb574b2,0x25837a58dc0921bd,0xd19113f97ca92ff6,
                                0x9432477322f54701,0x3ae5e58137c2dadc,0xc8b576349af3dda7,0xa94461460fd0030e,0xecc8c73ea4751e41,0xe238cd993bea0e2f,
                                0x3280bba1183eb331,0x4e548b384f6db908,0x6f420d03f60a04bf,0x2cb8129024977c79,0x5679b072bcaf89af,0xde9a771fd9930810,
                                0xb38bae12dccf3f2e,0x5512721f2e6b7124,0x501adde69f84cd87,0x7a5847187408da17,0xbc9f9abce94b7d8c,0xec7aec3adb851dfa,
                                0x63094366c464c3d2,0xef1c18473215d908,0xdd433b3724c2ba16,0x12a14d432a65c451,0x50940002133ae4dd,0x71dff89e10314e55,
                                0x81ac77d65f11199b,0x043556f1d7a3c76b,0x3c11183b5924a509,0xf28fe6ed97f1fbfa,0x9ebabf2c1e153c6e,0x86e34570eae96fb1,
                                0x860e5e0a5a3e2ab3,0x771fe71c4e3d06fa,0x2965dcb999e71d0f,0x803e89d65266c825,0x2e4cc9789c10b36a,0xc6150eba94e2ea78,
                                0xa5fc3c531e0a2df4,0xf2f74ea7361d2b3d]
        constant_table = []
        for r in range(self.nbr_rounds):
            bin_str_1 = format(constants[3 * r], "064b")
            bin_str_2 = format(constants[3 * r + 1], "064b")
            bin_str_3 = format(constants[3 * r + 2], "064b")
            constant_table.append([int(b) for b in bin_str_1] + [int(b) for b in bin_str_2] + [int(b) for b in bin_str_3])
        return constant_table
    
    
    def gen_test_vectors(self, version):
        # test vectors
        hex_str = 'B48F3216AB33AE0199142F6A0743E8481BFC37625CBBDC4F'
        
        if version == 192 and self.nbr_rounds==7:
            plaintext = [0 for _ in range(192)]
            ciphertext = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
            # ciphertext = '247D3080D263F74CB03DDE6E575C68EE68EEE957E1C29C50'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext = [int(b) for b in bin_str]
            
            # plaintext =  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
            # ciphertext =  [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
            # plaintext = '0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF'
            # bin_str = bin(int(plaintext, 16))[2:].zfill(192)
            # plaintext = [int(b) for b in bin_str]
            # ciphertext = 'B48F3216AB33AE0199142F6A0743E8481BFC37625CBBDC4F'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext =[int(b) for b in bin_str]
            return [[plaintext], ciphertext]
            
            
        elif version == 192 and self.nbr_rounds==6:
            plaintext = [0 for _ in range(192)]
            ciphertext = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
            # ciphertext = 'A6D518A2E57375151593110A161ED7C6278ABCD031CBE86C'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext = [int(b) for b in bin_str]
            
            
            # plaintext =  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
            # ciphertext = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
            # plaintext = '0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF'
            # bin_str = bin(int(plaintext, 16))[2:].zfill(192)
            # plaintext = [int(b) for b in bin_str]
            # ciphertext = 'CB4411341FFFB30003001A8C1F06FED87FF689C52D1EAB65'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext =[int(b) for b in bin_str]
            return [[plaintext], ciphertext]
            

        elif version == 192 and self.nbr_rounds==5:
            plaintext = [0 for _ in range(192)]
            ciphertext =  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
            # ciphertext = 'E0D56FBD9556A871CA49357A822D0481A8502DDD16FECE0F'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext = [int(b) for b in bin_str]
            
            # plaintext =  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
            # ciphertext =  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
            # plaintext = '0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF'
            # bin_str = bin(int(plaintext, 16))[2:].zfill(192)
            # plaintext = [int(b) for b in bin_str]
            # ciphertext = '123A5D7AD45DE44A27640BEF01F48D42017CFAD0F2223C3C'
            # bin_str = bin(int(ciphertext, 16))[2:].zfill(192)
            # ciphertext =[int(b) for b in bin_str]
            
            return [[plaintext], ciphertext]
        

def SPEEDY_PERMUTATION(r=None): 
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(192)], [var.Variable(1,ID="out"+str(i)) for i in range(192)]
    my_cipher = SPEEDY_permutation(f"SPEEDY_PERM", 192, my_input, my_output, nbr_rounds=r)
    return my_cipher