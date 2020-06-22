# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:15:13 2019

@author: cyaba
"""

from gurobipy import *
import numpy as np
from math import exp

class SP:
    def __init__(self, setting, budget, additional_budget, opening_cost, maintaining_cost, N, districts, initial_infected, regression1, regression2, s_shape1, s_shape2, probabilities, tau, T ):
        self.setting = setting
        self.budget = budget #budget for first stage
        self.additional_budget = additional_budget #additional budget for the second stage
        self.opening_cost = opening_cost #opening cost of ETUs per district (array)
        self.maintaining_cost = maintaining_cost #maintaining cost of ETUs per bed per district (array)
        self.N = N #number of districts
        self.districts = districts #dictionary for districts and their index
        self.initial_infected = initial_infected
        self.regression1 = regression1 #regression parameters for K1 for the s-shaped curve (2-d array)
        self.regression2 = regression2 #regression parameters for K2 for the s-shaped curve (2-d array)
        self.s_shape1 = s_shape1 #a1 and b1 for the s-shaped curve (array of tuples)
        self.s_shape2 = s_shape2 #a2 and b2 for the s-shaped curve (array of tuples)
        self.probabilities = probabilities #probability of the random additional budget
        self.tau = tau #time until the first stage ends
        self.T = T #end of planning horizon



    def stochastic_model(self):
        model = Model("Facilty_Location")

        # define decision variables
        # first stage
        m = {}
        I_tau = {}
        y = {}
        #second stage
        m_low = {}
        m_med = {}
        m_high = {}
        I_T_low = {}
        I_T_med = {}
        I_T_high = {}

        for n in range(0, self.N):
            #first stage variables
            m[n] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "capacity_"+self.districts[n])
            y[n] = model.addVar(vtype = GRB.BINARY, name = "opening_"+self.districts[n])
            I_tau[n] = model.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = "infected_tau"+self.districts[n])

            #second stage variables
            m_low[n] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "capacity_low_"+self.districts[n])
            m_med[n] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "capacity_med_"+self.districts[n])
            m_high[n] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "capacity_high_"+self.districts[n])
            I_T_low[n] = model.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = "infected_T_low"+self.districts[n])
            I_T_med[n] = model.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = "infected_T_med"+self.districts[n])
            I_T_high[n] = model.addVar(vtype = GRB.CONTINUOUS, lb = 0, name = "infected_T_high"+self.districts[n])

        model.update()
        #define the objective function
        second_stage_obj = quicksum(self.probabilities[0] * I_T_low[n] + self.probabilities[1] * I_T_med[n] + self.probabilities[2] * I_T_high[n] for n in range(0, self.N))
        model.setObjective(second_stage_obj, GRB.MINIMIZE)
        model.update()
        #define the constraints
        big_M = 100000
        #first stage
        #budget constraint
        budget_lhs = quicksum(self.opening_cost[n] * y[n] + self.maintaining_cost[n] * m[n] for n in range(0, self.N))
        model.addConstr(budget_lhs <= self.budget, name = "budget")

        #second stage budget constraint
        budget_low_lhs = quicksum(self.maintaining_cost[n] * (m_low[n]) for n in range(0, self.N))
        budget_med_lhs = quicksum(self.maintaining_cost[n] * (m_med[n]) for n in range(0, self.N))
        budget_high_lhs = quicksum(self.maintaining_cost[n] * (m_high[n]) for n in range(0, self.N))
        model.addConstr(budget_low_lhs <= self.additional_budget[0], name = "budget_low")
        model.addConstr(budget_med_lhs <= self.additional_budget[1], name = "budget_med")
        model.addConstr(budget_high_lhs <= self.additional_budget[2], name = "budget_high")


        for n in range(0, self.N):
            #first stage
            #opening constraint
            model.addConstr(m[n] <= big_M * y[n], name = "open_"+self.districts[n])

            #cumulative infected at time tau
            K1 = self.regression1[n][0] + quicksum(self.regression1[n][nPrime+1] * self.initial_infected[nPrime] + self.regression1[n][12+nPrime] * m[nPrime] for nPrime in range(0, self.N))
            a1, b1 = self.s_shape1[n]
            i_tau = K1 / (exp(a1*self.tau + b1))
            model.addConstr(I_tau[n], GRB.EQUAL, i_tau, name = "infected_tau"+self.districts[n])

            #second stage
            #opening constraint
            model.addConstr(m_low[n] <= big_M * y[n], name = "open_low_"+self.districts[n])
            model.addConstr(m_med[n] <= big_M * y[n], name = "open_med_"+self.districts[n])
            model.addConstr(m_high[n] <= big_M * y[n], name = "open_high_"+self.districts[n])

            #cumulative infected at time T
            K2_low = self.regression2[n][0] + quicksum(self.regression2[n][nPrime+1] * I_tau[n] + self.regression2[n][12+nPrime] * (m[nPrime] + m_low[nPrime]) for nPrime in range(0, self.N))
            K2_med = self.regression2[n][0] + quicksum(self.regression2[n][nPrime+1] * I_tau[n] + self.regression2[n][12+nPrime] * (m[nPrime] + m_med[nPrime]) for nPrime in range(0, self.N))
            K2_high = self.regression2[n][0] + quicksum(self.regression2[n][nPrime+1] * I_tau[n] + self.regression2[n][12+nPrime] * (m[nPrime] + m_high[nPrime]) for nPrime in range(0, self.N))
            a2, b2 = self.s_shape2[n]
            iT_low =K2_low / (exp(a2*(self.T - self.tau) + b2))
            iT_med =K2_med / (exp(a2*(self.T - self.tau) + b2))
            iT_high =K2_high / (exp(a2*(self.T - self.tau) + b2))

            model.addConstr(I_T_low[n], GRB.EQUAL, iT_low, name = "infected_T_low_"+self.districts[n])
            model.addConstr(I_T_med[n], GRB.EQUAL, iT_med, name = "infected_T_med_"+self.districts[n])
            model.addConstr(I_T_high[n], GRB.EQUAL, iT_high, name = "infected_T_high_"+self.districts[n])

            #bounds on cumulative infected
            #model.addConstr(I_tau[n] >= self.initial_infected[n], name = "bound_tau"+self.districts[n])
            #model.addConstr(I_T_low[n] >= I_tau[n], name='bound_T_low'+self.districts[n])
            #model.addConstr(I_T_med[n] >= I_tau[n], name='bound_T_med' + self.districts[n])
            #model.addConstr(I_T_high[n] >= I_tau[n], name='bound_T_high' + self.districts[n])

        model.update()

        model.optimize()
        solution_file = open("solution_"+str(self.setting)+".txt", "w")

        for v in model.getVars():
            solution_file.write(v.varName + ": " + str(v.X) + "\n")

        solution_file.write("Objective: "+str(model.objVal))

        solution_file.close()
    
    

      
      
      
      
      
    
    
    