## Given the log file path, extract the data
## from the file by the labels:
## "CONTROLLER_INPUT_STATES"
## "CONTROLLER_OUTPUT"
## Tao Chen, 05/21/2018

import sys
import csv

STATE_STR = "CONTROLLER_INPUT_STATES"
OUTPUT_STR = "CONTROLLER_OUTPUT"
is_state = False
is_output = False
states = []
outputs = []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_data.py [path]")
    else:
        data_path = sys.argv[1] + "/rosout.log"
        f = open(data_path, 'r')
        counter = 1
        for line in f:
            # if counter%5 == 0:  ## reduce the frequency
            #     counter += 1
            #     continue
            ## test for state
            index = line.find(STATE_STR)
            if index != -1:
                index = index + len(STATE_STR)+2
                states.append([float(i) for i in line[index:-2].split(' ')])
                counter += 1
                continue
            ## test for output
            index = line.find(OUTPUT_STR)
            if index != -1:
                index = index + len(OUTPUT_STR)+2
                outputs.append([float(i) for i in line[index:-2].split(' ')])
                counter += 1

    ## make sure lengths are the same
    print(len(states), len(outputs))
    if len(states) != len(outputs):
        if len(states) > len(outputs):
            states = states[:len(outputs)]
        else:
            outputs = outputs[:len(states)]

    ## get rid of the starting 0's
    num_of_zeros = 0
    for i in range(len(outputs)):
        if outputs[i][0] == 0:
            num_of_zeros += 1
        else:
            break
    states = states[num_of_zeros:]
    outputs = outputs[num_of_zeros:]

    statesFile = open('controller_input.csv', 'w')
    controlFile = open('controller_output.csv', 'w')

    with statesFile:
        writer = csv.writer(statesFile)
        writer.writerows(states)
        statesFile.close()

    with controlFile:
        writer = csv.writer(controlFile)
        writer.writerows(outputs)
        controlFile.close()
