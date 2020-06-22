# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:14:18 2019

@author: cyaba
"""

import stochastic_programming
import numpy as np
from operator import add


def get_districts():
    districts = np.loadtxt("parameters/districts.txt", dtype='str', delimiter='\t')
    dist = {}
    for i in districts:
        dist[int(i[0])] = i[1]
    return dist


def get_s_shaped_params(filename):
    # saves as list of tuples
    parameters = np.loadtxt(filename, dtype="d", delimiter=",")
    retVal = []
    for i in parameters:
        retVal.append((i[0], i[1]))

    return retVal


def main():
    # parameters used in stochastic program
    N = 11
    tau = 60
    T = 188
    districts = get_districts()
    budget = 505000
    settings = [20, 60, 150]  # number of days used in different settings to estimate parameters for the s-shaped curves
    additional_budget = [235000, 470000, 705000]
    probabilities = [0.25, 0.5, 0.25]
    opening_cost = np.loadtxt("parameters/opening_cost.txt", delimiter=",", dtype="d")
    medical_cost = np.loadtxt("parameters/medical_cost.txt", delimiter=",", dtype="d")
    transportation_cost = np.loadtxt("parameters/transportation_cost.txt", delimiter=",", dtype="d")
    maintaining_cost = list(map(add, medical_cost, transportation_cost))
    for s in settings:
        initial_infected = np.loadtxt("parameters/initial_infected.txt", delimiter=",", dtype="i")
        regression1 = np.loadtxt("parameters/regression_params1_" + str(s) + ".txt", delimiter=",", dtype="d")
        regression2 = np.loadtxt("parameters/regression_params2_" + str(s) + ".txt", delimiter=",", dtype="d")
        s_shape1 = get_s_shaped_params("parameters/initial_parameters1_" + str(s) + ".txt")
        s_shape2 = get_s_shaped_params("parameters/initial_parameters2_" + str(s) + ".txt")

        stochastic_ip = stochastic_programming.SP(s, budget, additional_budget, opening_cost, maintaining_cost, N,
                                                  districts, initial_infected, regression1, regression2, s_shape1,
                                                  s_shape2, probabilities, tau, T)

        stochastic_ip.stochastic_model()


if __name__ == '__main__':
    main()
