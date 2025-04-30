import copy
import primitives.primitives as prim

def generate_copy_added_representation(cipher):
    """
    Generate a new cipher instance with added copy layer after each operator.

    Args:
        cipher (object): The cipher input instance.
    Returns:
        cipher (object): The cipher output with added copy layers.
    """

    new_cipher = copy.deepcopy(cipher)
    for i in range(len(cipher.states)): 
        new_state = prim.State(cipher.states[i].name, cipher.states[i].size, cipher.states[i].wordsize, cipher.states[i].nbr_rounds, 2*cipher.states[i].layers) 
        for r in range(1,cipher.states[i].nbr_rounds+1):
            for j in range(cipher.states[i].layers):
                new_state.layers[2*j] = cipher.states[i].layers[j]
                new_state.AddIdentityLayer("COPY", r, 2*j+1)  

        new_cipher.states[i] = new_state
