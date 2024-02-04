from ast import List
from main import analyse_image
import cv2
import argparse

def error():
    print("error!")
    return

nesting_stack = []

def main(img, input):

    def get_digits(start_pos):
        p_pos = start_pos
        while(processed_image[p_pos].isdigit()):
            p_pos += 1
        if start_pos == p_pos:
            error() #expected number, instead got something else
            print("here 1")
        else:
            return int("".join(processed_image[start_pos:p_pos])), p_pos - 1
    
    def create_program_string(current_pos, end_tok_pos):
        program_string = ""
        while current_pos < end_tok_pos:
            tok = processed_image[current_pos]
            # accumulator update functions
            if tok == "add":
                program_string = program_string + ("+ ")
            elif tok == "sub":
                program_string = program_string + ("- ")
            elif tok == "mult":
                program_string = program_string + ("x ")
            elif tok == "div":
                program_string = program_string + ("÷ ")
            elif tok == "=":
                program_string = program_string + ("= ")
            elif tok == "not":
                program_string = program_string + ("not ")
            elif tok == "and":
                program_string = program_string + ("and ")
            elif tok == "or":
                program_string = program_string + ("or ")
            elif tok == "if":
                program_string = program_string + ("if ")
            elif tok == "then":
                program_string = program_string + ("then")
            elif tok == "else":
                program_string = program_string + ("else")
            elif tok == "end":
                program_string = program_string + ("end")
            elif tok == "newline":
                program_string = program_string + ("\n")
            elif tok == "repeat":
                rep_num, current_pos = get_digits(current_pos + 1)
                program_string = program_string + ("repeat " + str(rep_num) + ":")
            elif tok == "while":
                program_string = program_string + ("while ")
            elif tok == "do":
                program_string = program_string + ("do")
            elif tok == "print":
                program_string = program_string + ("print ")
            elif tok == "divisible":
                program_string = program_string + ("divisible ")
            elif tok == "gt":
                program_string = program_string + ("> ")
            elif tok == "lt":
                program_string = program_string + ("< ")
            elif tok.isdigit():
                program_string = program_string + (tok) + " "
            else:
                error()
                print("here2")
                print(tok)
            current_pos += 1
        return program_string

    def execute_code(curr_pos, acc, sys_out):

        def locate_next_instance_of(token):
            pos = curr_pos           
            while not processed_image[pos] == token:
                pos += 1
                if (pos >= image_len):
                    return -1
            return pos

        def evaluate_condition(start_pos, then_pos):
            if start_pos >= then_pos:
                error()
                print("here3")
            if processed_image[start_pos] == "not":
                eval, end_pos = evaluate_condition(start_pos + 1, then_pos)
                return not eval, end_pos
            elif processed_image[start_pos] == "divisible":
                digits, new_pos = get_digits(start_pos + 1)
                return acc % digits == 0, new_pos + 1
            elif processed_image[start_pos] == "lt":
                digits, new_pos = get_digits(start_pos + 1)
                return (acc < digits), new_pos + 1
            elif processed_image[start_pos] == "gt":
                digits, new_pos = get_digits(start_pos + 1)
                return (acc > digits), new_pos + 1
            else:
                error()
                print("here4")

        # def evaluate_string(start_posit, end_posit):
        #     #TODO
        #     error()
        #     print("here5")

        while curr_pos < image_len:
            tok = processed_image[curr_pos]
            # accumulator update functions
            if tok == "add":
                acc_increment, curr_pos = get_digits(curr_pos + 1)
                acc += acc_increment
            elif tok == "sub":
                acc_decrement, curr_pos = get_digits(curr_pos + 1)
                acc -= acc_decrement
            elif tok == "mult":
                acc_mult, curr_pos = get_digits(curr_pos + 1)
                acc *= acc_mult
            elif tok == "div":
                acc_divisor, curr_pos = get_digits(curr_pos + 1)
                acc /= float(acc_divisor)
            # complex functions
            elif tok == "if":
                (cond, curr_pos) = evaluate_condition(curr_pos + 1, locate_next_instance_of("then"))
                if processed_image[curr_pos] != "then":
                    error()
                    print("here14")
                    print(curr_pos)
                if (cond):
                    acc, sys_out, curr_pos = execute_code(curr_pos + 1, acc, sys_out)
                    if processed_image[curr_pos] == "else":
                        _, _, curr_pos = execute_code(curr_pos + 1, acc, sys_out)
                else:
                    _, _, curr_pos = execute_code(curr_pos + 1, acc, sys_out)
                    if processed_image[curr_pos] == "else":
                        acc, sys_out, curr_pos = execute_code(curr_pos + 1, acc, sys_out)
                        if (curr_pos != "end"):
                            error() #end          
            elif tok == "end":
                return acc, sys_out, curr_pos
            elif tok == "else":
                return acc, sys_out, curr_pos
            elif tok == "repeat":
                rep_num, curr_pos = get_digits(curr_pos + 1)
                for _ in range (rep_num):
                    acc, sys_out, loop_end_pos = execute_code(curr_pos + 1, acc, sys_out)
                curr_pos = loop_end_pos
            elif tok == "while":
                cond_start = curr_pos + 1
                cond, _ = evaluate_condition(cond_start, curr_pos)
                curr_pos = locate_next_instance_of("do")
                if curr_pos == -1:
                    error()
                    print("here9")
                while (cond):
                    (acc, sys_out, loop_end) = execute_code(curr_pos + 1, acc, sys_out)
                    cond, _ = evaluate_condition(cond_start, curr_pos) 
                    curr_pos = loop_end
            elif tok == "print":
                start_to_print = curr_pos + 1
                curr_pos = locate_next_instance_of("newline")
                if curr_pos == -1:
                    error()
                    print("here11")
                elif (curr_pos == start_to_print):
                    sys_out = sys_out + str(acc) + "\n"
                # else:
                #     (new_sys_out) = evaluate_string(start_to_print, curr_pos)
                #     print(new_sys_out)
                #     sys_out = sys_out + new_sys_out + "\n"
            else:
                if tok != "newline":
                    error()
                    print(tok)
                    print("here13")
            curr_pos += 1
        return acc, sys_out, curr_pos
    
    def convert_tokens(toks):
        converted_lines = []
        for line in toks:
            converted_lines.extend(line)
            converted_lines.append('newline')

        converted_lines.append('end')
        # print(converted_lines)
        # exit()

        return converted_lines
        
    
    processed_image = convert_tokens(analyse_image(img))
    accumulator_update_functions = List(["add", "sub", "mult", "div"])
    boolean_functions = List(["divisible", "=", "not", "<", ">"])
    acc = input
    sys_out = ""
    image_len = len(processed_image)
    program_string = create_program_string(0, image_len)
    acc, sys_out, _ = execute_code(0, acc, sys_out)
    return program_string, acc, sys_out

def evaluate_submission(image_path, test_inputs):
    image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/flash.jpg')
    image = cv2.imread(image_path)
    to_return = ""
    for input in test_inputs.split():
        program_string, result, output = main(image, int(input))
        to_return += "\n" + input + "\n\n" + str(result) + "\n" + output
    to_return += "\n" + program_string
    return to_return

#string

#input1
#output1

#sysout1
        

