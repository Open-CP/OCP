def _mobius_transform_in_place(table, bit_size): # Compute the Mobius transform in place to convert a truth table into ANF coefficients.
    for i in range(bit_size):
        step = 2 ** (bit_size - 1 - i)
        block = 2 ** (bit_size - i)
        for j in range(2 ** i):
            for k in range(step):
                table[k + step + j * block] ^= table[k + j * block]


def anf_support_terms(sbox_table, bit_size): # Compute ANF support terms for every nonzero output mask of an S-box.
    supports = [[] for _ in range(2 ** bit_size)]
    for output_mask in range(1, 2 ** bit_size):
        table = [1 if (output_mask & value) == output_mask else 0 for value in sbox_table]
        _mobius_transform_in_place(table, bit_size)
        supports[output_mask] = [term for term, coefficient in enumerate(table) if coefficient != 0]
    return supports


def sbox_two_subset_division_trails(sbox_table, bit_size): # Generate bit-based two-subset division trails for an S-box from its ANF support.
    anf_terms = anf_support_terms(sbox_table, bit_size)
    trails = [[0 for _ in range(2 * bit_size)]]

    for input_mask in range(1, 2 ** bit_size):
        minimal_outputs = []
        for output_mask in range(1, 2 ** bit_size):
            if not any((input_mask | term) == term for term in anf_terms[output_mask]):
                continue

            dominated_outputs = []
            should_add = True
            for previous_output in minimal_outputs:
                if (previous_output | output_mask) == output_mask:
                    should_add = False
                    break
                if (previous_output | output_mask) == previous_output:
                    dominated_outputs.append(previous_output)

            if should_add:
                for previous_output in dominated_outputs:
                    minimal_outputs.remove(previous_output)
                minimal_outputs.append(output_mask)

        for output_mask in minimal_outputs:
            trails.append(
                [int(bit) for bit in format(input_mask, f"0{bit_size}b")]
                + [int(bit) for bit in format(output_mask, f"0{bit_size}b")]
            )
    return trails


def trails_to_truthtable(trails, width): # Convert a list of binary trails to a truth table string.
    ttable = ["0"] * (2 ** width)
    for trail in trails:
        if len(trail) != width:
            raise ValueError(f"trail width {len(trail)} does not match expected width {width}")
        index = int("".join(str(bit) for bit in trail), 2)
        ttable[index] = "1"
    return "".join(ttable)


def two_subset_sbox_truthtable(sbox_table, bit_size): # Generate the truth table of bit-based two-subset division trails for an S-box.
    trails = sbox_two_subset_division_trails(sbox_table, bit_size)
    return trails_to_truthtable(trails, 2 * bit_size)
