#!/usr/bin/env python

import sys, getopt
import learner


def main(model_file, hidden_sizes, num_iterations, learning_rate):
    l = learner.Learner(model_file, hidden_sizes, num_iterations, learning_rate)

    l.learn()
    l.exploit()


def help():
    print('test.py -f model file -i num_iterations -l learning_rate')


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:i:l:s:", ["file=,hidden="])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    num_iterations = 100
    model_file = "model.pt"
    learning_rate = 0.01
    hidden_sizes = []

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt in ("-f", "--file"):
            model_file = arg
        elif opt in ("-s", "--hidden"):
            hidden_sizes.append(int(arg))
        elif opt in ("-i"):
            num_iterations = int(arg)
        elif opt in ("-l"):
            learning_rate = float(arg)

    if len(hidden_sizes) == 0:
        hidden_sizes = [48, 96, 128, 64, 16]
    main(model_file, hidden_sizes, num_iterations, learning_rate)
