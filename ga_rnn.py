# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:55:12 2018

@author: Christian
"""
# IMPORT PACKAGES ######################################################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from matplotlib import pyplot as plt


# Simulate data
time_steps = np.array(range(0, 8000))
observed = np.array(np.sin(time_steps) + np.random.uniform(-1.5, 1.5) +
                    10 * np.sin(time_steps / (100 + np.random.uniform(-5.5, 5.5))) +
                    3 * np.sin(time_steps / (20 + np.random.uniform(-5.5, 5.5))) + np.random.uniform(-15.5, 1.5))

calibrate = np.array(np.sin(time_steps))
my_data = np.reshape(np.array(observed), (len(time_steps), 1))
train_data = my_data[0:6250]
test_data = my_data[6250:]


# GLOBAL VARIABLES #####################################################################################################
# Genetic Algorithm
population_size = 5
num_generations = 5
# crossover points
cx_1 = 4
cx_2 = 5 + cx_1
# gene_length = 2 + cx_2
cx_3 = 2 + cx_2
gene_length = 2 + cx_3

# Define Key Objects
forecast = 21
dropout_dict = {0: .4, 1: .3, 2: .2, 3: .1}
activation_dict = {0: 'tanh', 1: 'linear', 2: 'sigmoid', 3: 'relu'}
logbook = []
count = []


# Function to resize data. Will turn single column vector into 3d tensor with
# dimensions according to the window_size, Forecast_size and data length
def prepare_data(data, window_size, forecast_size):
    X, Y = np.empty((0, window_size)), np.empty((0, forecast_size))
    for i in range(len(data) - window_size - forecast_size):
        X = np.vstack([X, data[i:(i + window_size), 0]])
        Y = np.vstack([Y, data[(i + window_size):(i + window_size + forecast_size), 0]])
    X = np.reshape(X, (len(X), window_size, 1))
    Y = np.reshape(Y, (len(Y), forecast_size))
    return X, Y


# train an individual model and assign a fitness value
def train_evaluate(ga_individual_solution, forecast_size):
    # make a counter
    count.append(1)
    # Decode GA solution to integer for window_size and num_units
    window_size = BitArray(ga_individual_solution[:cx_1]).uint
    num_units = BitArray(ga_individual_solution[cx_1:cx_2]).uint
    dropout_key = BitArray(ga_individual_solution[cx_2:cx_3]).uint
    dropout_rate = dropout_dict[dropout_key]
    activation_key = BitArray(ga_individual_solution[cx_3:]).uint
    activation = activation_dict[activation_key]
    # Set minimum sizes for window and units to 1
    if window_size == 0:
        ga_individual_solution[:cx_1] = [0] * (cx_1 - 1) + [1]
        window_size = 1
    if num_units == 0:
        ga_individual_solution[cx_1:cx_2] = [0] * (cx_2 - cx_1 - 1) + [1]
        num_units = 1
    if num_units == 0:
        ga_individual_solution[cx_2:cx_3] = [0] * (cx_3 - cx_2 - 1) + [1]
        num_units = 1

    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units, ', Dropout Rate: ', dropout_rate)
    print('Genome: ', window_size, num_units, dropout_key, 'count: ',
          len(count), 'activation: ', activation)

    # Reshape data based on window size and forecast size.
    X, Y = prepare_data(train_data, window_size, forecast)
    X_train, X_val, Y_train, Y_val = split(X, Y, test_size=0.20)

    # LSTM Model, uses a window of previous data to predict some forecast length.
    # e.g. I have the five previous days data and I want to predict the next week
    model_individual = Sequential()
    model_individual.add(LSTM(num_units, input_shape=(window_size, 1)))
    model_individual.add(Dropout(dropout_dict[dropout_key]))
    model_individual.add(Dense(forecast_size, activation=activation))
    model_individual.compile(optimizer='rmsprop', loss='mean_squared_error')
    history = model_individual.fit(X_train, Y_train, epochs=30, batch_size=10, shuffle=True)
    Y_pred = model_individual.predict(X_val)
    model_individual.count_params()
    # Loss Plot
    plt.plot(history.history['loss'], label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(list(range((population_size * num_generations) + population_size)))
    plt.title('Training Performance of Individuals')

    # Calculate fitness
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
    N = num_units + window_size + 1
    M = np.power(2, cx_2 - cx_1) + np.power(2, cx_1) + 1
    C = (window_size * num_units) + num_units
    CM = (np.power(2, cx_1) * np.power(2, cx_2 - cx_1)) + cx_2 - cx_1
    alpha = 3
    penalty = alpha * (((N / M) + (C / CM)) / 2)
    adjusted_rmse = rmse + penalty
    # Append individual profile to logbook
    logbook.append([len(count), window_size, num_units, dropout_dict[dropout_key],
                    rmse, adjusted_rmse, activation])
    model_individual.summary()
    print('\nValidation RMSE: ', rmse, 'adjusted: ', adjusted_rmse)
    return adjusted_rmse,


def trait_swap(ind1, ind2):
    trait1_swap = bernoulli.rvs(.5)
    trait2_swap = bernoulli.rvs(.5)
    trait3_swap = bernoulli.rvs(.5)
    trait4_swap = bernoulli.rvs(.5)
    if trait1_swap == 1:
        ind1[:cx_1], ind2[:cx_1], = ind2[:cx_1], ind1[:cx_1]
    if trait2_swap == 1:
        ind1[cx_1:cx_2], ind2[cx_1:cx_2], = ind2[cx_1:cx_2], ind1[cx_1:cx_2]
    if trait3_swap == 1:
        ind1[cx_2:], ind2[cx_2:], = ind2[cx_2:], ind1[cx_2:]
    if trait4_swap == 1:
        ind1[cx_3:], ind2[cx_3:], = ind2[cx_3:], ind1[cx_3:]
    return ind1, ind2


# GENETIC ALGORITHM OBJECTS ############################################################################################
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary,
                 n=gene_length)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.3)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('evaluate', train_evaluate, forecast_size=forecast)

population = toolbox.population(n=population_size)
algorithms.eaSimple(population, toolbox, cxpb=1, mutpb=0.5,
                    ngen=num_generations, verbose=True)

# OUTPUT AND PLOTS ##################################################################################################
# Data Summary Statistics
observed.min()
observed.max()
np.std(observed)

# Plot data zoomed in to see small waves
plt.plot(time_steps[:50], observed[:50], marker='o')
plt.xlabel('Timestep')
plt.ylabel('Observed Value')
plt.title('First 50 Data points')

# Plot data
plt.plot(time_steps[:100], observed[:100], marker='o')
plt.xlabel('Timestep')
plt.ylabel('Observed Value')
plt.title('First 100 Data points')

# Plot data zoomed out to see macro waves
plt.plot(time_steps[:500], observed[:500])
plt.xlabel('Timestep')
plt.ylabel('Observed Value')
plt.title('First 500 Data points')

# Plot data zoomed out to see macro waves
plt.plot(time_steps[:5000], observed[:5000])
plt.xlabel('Timestep')
plt.ylabel('Observed Value')
plt.title('First 5000 Data points')

# Plot confirmation data
plt.plot(time_steps[:100], calibrate[:100])
plt.xlabel('Timestep')
plt.ylabel('Observed Value')
plt.title('First 100 Calibration Data points')

# Generate sorted records of individual performances
df = pd.concat([pd.Series(x) for x in logbook], axis=1).transpose()
df.columns = ['individual', 'num input nodes', 'num hidden nodes',
              'dropout rate', 'rmse', 'adjusted rmse']
# Sort by adjusted
df.sort_values(by=['adjusted rmse'])


# Plot Model Node Config
x_ticks = range(np.power(2, cx_1))
y_ticks = range(np.power(2, cx_2 - cx_1))
plt.scatter(df['num input nodes'], df['num hidden nodes'],
            s=200 * df['rmse'], c=df['adjusted rmse'], marker='s')
plt.legend(['Individual'])
plt.xlabel('number of input nodes')
plt.ylabel('number of hidden nodes')
plt.grid(b=True)
plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.title('Model Performance During Training')
plt.colorbar()