from primitives.primitives import Stream_cipher
from operators.boolean_operators import XOR, AND
from operators.operators import Equal
import variables.variables as var


# Trivium Stream Cipher
# Based on the specification from: https://en.wikipedia.org/wiki/Trivium_(cipher)
# Trivium has a 288-bit internal state consisting of three shift registers
# Key: 80 bits, IV: 80 bits (can be less)
# Initialization: 1152 rounds (4 * 288)
# Output generation: up to 2^64 bits

class Trivium_Stream_Cipher(Stream_cipher):
    def __init__(self, name, iv_input, k_input, keystream_output, nbr_rounds_init=1152, nbr_rounds_update=1, nbr_rounds_keystream=1, represent_mode=0):
        """
        Initialize the Trivium stream cipher
        :param name: Name of the cipher
        :param iv_input: Input IV (80 bits as 80 1-bit variables)
        :param k_input: Input Key (80 bits as 80 1-bit variables)
        :param keystream_output: Output keystream
        :param nbr_rounds_init: Number of initialization rounds (default: 1152)
        :param nbr_rounds_update: Number of state update rounds (default: 1)
        :param nbr_rounds_keystream: Number of keystream generation rounds (default: 1)
        :param represent_mode: Integer specifying the mode of representation
        """
        
        if represent_mode == 0:
            # Trivium state: 288 bits split into three registers
            # Register A: 93 bits
            # Register B: 84 bits  
            # Register C: 111 bits
            # Total: 93 + 84 + 111 = 288 bits
            
            # Configuration for initialization function
            # Input: IV (80 bits) + Key (80 bits) = 160 bits
            # State: 288 bits (initialized with IV, Key, and fixed pattern)
            init_nbr_layers = 1
            init_nbr_words = 288  # Internal state size
            init_nbr_temp_words = 0
            init_word_bitsize = 1  # Bit-level operations
            
            # Configuration for state update function
            # Takes 288-bit state and produces updated 288-bit state
            update_nbr_layers = 5  # Multiple operations per round
            update_nbr_words = 288
            update_nbr_temp_words = 6  # Temporary variables for intermediate XOR/AND results
            update_word_bitsize = 1
            
            # Configuration for keystream generation function
            # Takes 288-bit state and produces keystream bits
            keystream_nbr_layers = 1
            keystream_nbr_words = 1  # Output keystream bit(s)
            keystream_nbr_temp_words = 0
            keystream_word_bitsize = 1
            
            super().__init__(name, iv_input, k_input, keystream_output, 
                           nbr_rounds_init, nbr_rounds_update, nbr_rounds_keystream,
                           [init_nbr_layers, init_nbr_words, init_nbr_temp_words, init_word_bitsize],
                           [update_nbr_layers, update_nbr_words, update_nbr_temp_words, update_word_bitsize],
                           [keystream_nbr_layers, keystream_nbr_words, keystream_nbr_temp_words, keystream_word_bitsize])
            
            INIT = self.functions["INITIALIZATION"]
            UPDATE = self.functions["STATE_UPDATE"]
            KEYSTREAM = self.functions["KEYSTREAM_GEN"]
            
            # ==================== INITIALIZATION ====================
            # Initialize the 288-bit state with:
            # - Bits 0-79: Key (k0...k79)
            # - Bits 93-172: IV (v0...v79) 
            # - Bits 285-287: Three 1's (rest are 0's)
            # Register layout: [a0...a92, b0...b83, c0...c110]
            # a: bits 0-92 (93 bits)
            # b: bits 93-176 (84 bits)
            # c: bits 177-287 (111 bits)
            
            for i in range(1, nbr_rounds_init + 1):
                # The initialization just sets up the initial state
                # All actual work happens in the UPDATE function
                # This is a pass-through to maintain the state
                INIT.AddIdentityLayer("INIT_PASS", i, 0)
            
            # ==================== STATE UPDATE ====================
            # Trivium update equations (from Wikipedia):
            # t1 = a65 + a92 + (a90 & a91) + b77
            # t2 = b68 + b83 + (b81 & b82) + c86
            # t3 = c65 + c110 + (c108 & c109) + a68
            # Then shift and update:
            # a_new = [t3, a0, a1, ..., a91]  (shift right, insert t3 at position 0)
            # b_new = [t1, b0, b1, ..., b82]  (shift right, insert t1 at position 0)
            # c_new = [t2, c0, c1, ..., c109] (shift right, insert t2 at position 0)
            
            for i in range(1, nbr_rounds_update + 1):
                # Calculate t1 = a65 + a92 + (a90 & a91) + b77
                # Temp vars: [t1_xor1, t1_and, t1_xor2, t2_xor1, t2_and, t2_xor2]
                # Index mapping: a is 0-92, b is 93-176, c is 177-287
                
                # t1 component: a65 + a92
                UPDATE.SingleOperatorLayer("T1_XOR1", i, 0, XOR, [[65, 92]], [288])
                
                # t1 component: a90 & a91
                UPDATE.SingleOperatorLayer("T1_AND", i, 1, AND, [[90, 91]], [289])
                
                # t1 component: (a65 + a92) + (a90 & a91)
                UPDATE.SingleOperatorLayer("T1_XOR2", i, 2, XOR, [[288, 289]], [290])
                
                # t1 final: t1_xor2 + b77
                UPDATE.SingleOperatorLayer("T1_FINAL", i, 3, XOR, [[290, 93 + 77]], [291])  # b77 is at index 93+77=170
                
                # Calculate t2 = b68 + b83 + (b81 & b82) + c86
                # t2 component: b68 + b83
                UPDATE.SingleOperatorLayer("T2_XOR1", i, 4, XOR, [[93 + 68, 93 + 83]], [292])
                
                # TODO: Continue implementing the full update logic
                # For now, using identity to maintain structure
                for j in range(288):
                    if j not in []:  # Will update this list with processed indices
                        pass
                
                # Placeholder: maintain state through identity
                # In a complete implementation, we would shift the registers and insert t1, t2, t3
                UPDATE.AddIdentityLayer("UPDATE_PASS", i, 5)
            
            # ==================== KEYSTREAM GENERATION ====================
            # Output bit: z = a65 + a92 + b68 + b83 + c65 + c110
            # This is computed from the current state
            
            for i in range(1, nbr_rounds_keystream + 1):
                # For simplicity, extracting one bit of keystream
                # In practice, you'd compute: z = a65 + a92 + b68 + b83 + c65 + c110
                # For now, using a simplified pass-through
                KEYSTREAM.AddIdentityLayer("KEYSTREAM_GEN", i, 0)
    
    def gen_test_vectors(self):
        """
        Generate test vectors for Trivium
        Test vectors can be found in the eSTREAM documentation
        """
        # Example test vector (simplified - actual vectors would come from official sources)
        # Key: 80 zero bits, IV: 80 zero bits
        # Expected first keystream bits after initialization
        
        # For a proper implementation, add official test vectors here
        pass


def TRIVIUM(nbr_rounds_init=1152, nbr_rounds_update=1, nbr_rounds_keystream=1, represent_mode=0, copy_operator=False):
    """
    Create a Trivium stream cipher instance
    :param nbr_rounds_init: Number of initialization rounds (default: 1152)
    :param nbr_rounds_update: Number of state update rounds (default: 1)
    :param nbr_rounds_keystream: Number of keystream generation rounds (default: 1)
    :param represent_mode: Representation mode
    :param copy_operator: Whether to add copy operators
    :return: Trivium stream cipher instance
    """
    # Create IV input: 80 bits
    iv_input = [var.Variable(1, ID="iv" + str(i)) for i in range(80)]
    
    # Create Key input: 80 bits
    k_input = [var.Variable(1, ID="k" + str(i)) for i in range(80)]
    
    # Create Keystream output: variable size (start with 1 bit)
    keystream_output = [var.Variable(1, ID="ks" + str(i)) for i in range(1)]
    
    # Create Trivium instance
    my_cipher = Trivium_Stream_Cipher("TRIVIUM", iv_input, k_input, keystream_output,
                                      nbr_rounds_init=nbr_rounds_init,
                                      nbr_rounds_update=nbr_rounds_update,
                                      nbr_rounds_keystream=nbr_rounds_keystream,
                                      represent_mode=represent_mode)
    
    # Post-processing
    my_cipher.clean_graph()
    if copy_operator:
        my_cipher.add_copy_operators()
    my_cipher.build_dictionaries()
    my_cipher.gen_test_vectors()
    
    return my_cipher

