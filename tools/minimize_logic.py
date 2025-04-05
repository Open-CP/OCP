try:
    from pyeda.inter import espresso_tts
except ImportError:
    print("pyeda is not installed, installing it by 'pip3 install pyeda', https://pyeda.readthedocs.io/en/latest/")        
import subprocess
import os


def ttb_to_cnf_espresso(model_type, ttable, num_vars, variables, mode=0):    
    cont_ttable = ''
    for n in range(2**(num_vars)): 
        cont_ttable += f'{bin(n)[2:].zfill(num_vars)} {ttable[n]}\n' 
    file_contents = f".i {num_vars}\n"
    file_contents += ".o 1\n"
    file_contents += f".p {2**(num_vars)}\n"
    file_contents += ".ilb " + " ".join(variables) + "\n"     
    file_contents += ".ob F\n"
    file_contents += ".type fr\n" 
    file_contents += cont_ttable
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    base_path = os.path.join(script_dir, '..', 'files')
    base_path = os.path.abspath(base_path)
    filename_ttable = os.path.join(base_path, 'ttable.txt')
    filename_sttable = os.path.join(base_path, 'sttable.txt')
    with open(filename_ttable, "w") as fw:
        fw.write(file_contents)
        fw.write(".e\n")

    espresso_options =  [['-estrong', '-eonset'], [], ['-eonset']]
    espresso_command = ['espresso', *espresso_options[mode], filename_ttable] # Espresso Script of Pyeda provides the parameters: "-e {fast,ness,nirr,nunwrap,onset,strong}"
    result = subprocess.run(espresso_command, capture_output=True, text=True)
    espresso_output = result.stdout
    with open(filename_sttable, 'w') as fw: 
        fw.write(espresso_output)
    with open(filename_sttable, 'r') as fr: 
        espresso_output = fr.readlines()   

    constraints = []
    starting_point = 0
    end_point = 0
    for i in range(len(espresso_output)):
        if ".p" in espresso_output[i]: starting_point = i + 1
        if ".e" in espresso_output[i]: end_point = i
    for l in espresso_output[starting_point:end_point]:
        line = l[0:len(variables)]
        constraint = ''
        lp_rhs = 0
        for i in range(len(variables)):
            if model_type == 'milp':
                if line[i] == '0': constraint += " + " + variables[i]
                elif line[i] == '1': 
                    constraint += " - "+ variables[i]
                    lp_rhs += 1
            elif model_type == 'sat':
                if line[i] == '0': constraint += ' ' + variables[i]
                elif line[i] == '1': constraint += ' -' + variables[i]
        if model_type == 'milp':
            constraint = constraint.strip(" + ")
            constraint += ' >= ' + str(-(lp_rhs - 1))
        elif model_type == 'sat': constraint = constraint.strip(" ")
        constraints.append(constraint)
    return constraints
