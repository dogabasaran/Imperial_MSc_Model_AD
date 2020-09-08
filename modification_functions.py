import math
import numpy as np
import random
import matplotlib.pyplot as plt

# N = 1000
# nsecs = 1440
# dt = 0.1
# simtime = np.linspace(start=0, stop=nsecs-dt, num=nsecs/dt)
# # Example weight matrix
# wo = np.ones((1000, 1))
# M = np.ones((1000, 1000))


'''
4 sine waves
'''
# amp = 1.3
# freq = 1/720
# ft = (amp/1.0)*np.sin(1.0*(math.pi)*freq*simtime) + \
#      (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime) + \
#      (amp/6.0)*np.sin(3.0*(math.pi)*freq*simtime) + \
#      (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime)
# ft = ft/1.5



'''
multiple tophat functions:
'''
# noise = np.random.normal(0,.015,14400)
# amplitude = 2
# counter = 0
# counter2 = 0
# ft = np.zeros(nsecs*10)
# for idx, _ in enumerate(simtime):
#     if counter < 1000:
#         ft[idx] = 1
#         counter += 1
#     else:
#         ft[idx] = 0
#         counter2 += 1
#     if counter2 == 1000:
#         counter = 0
#         counter2 = 0
# ft = (ft - 0.5) * amplitude


# Get a random neuron from the network
def get_choice(how_many_choices, alter_around):
    choices = random.choices(np.linspace(0, 9999, 10000),
                             k=how_many_choices)
    choices = [int(choice) for choice in choices]
    for choice in choices:
        # If any of the choices after the +- range are out of bounds, make new
        # choices until they aren't.
        if choice-alter_around < 0 or choice + alter_around > 10000:
            choices = get_choice(how_many_choices, alter_around)
    return choices


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Not used
def apply_gaussian_readoutweights(wo, how_many_choices=1, alter_around=0,
                                filter_width=5, depth=0.5):
    orig_wo = wo
    choices = get_choice(how_many_choices, alter_around)
    x_values = np.linspace(0, int(len(wo))-1, int(len(wo)))
    for choice in choices:
        gauss_filter = 1 - depth * gaussian(x=x_values, mu=choice,
                                            sig=filter_width)
        gauss_filter = np.expand_dims(gauss_filter, axis=1)
        wo = np.multiply(wo, gauss_filter)
    # plt.figure(2, figsize=(20, 7))
    # plt.title("Readout weight alteration")
    # plt.plot(x_values, orig_wo, label="Original weights")
    # plt.plot(x_values, wo, label="Altered weights")
    # plt.legend()
    # plt.show()
    return wo


'''
remove random % of connections (weights)
'''


def remove_random_readoutweights(wo, alter=False, alter_percentage=0.03):
    if alter:
        choices = random.choices(np.linspace(0, len(wo)-1, len(wo)),
                                 k=int(alter_percentage*(len(wo))))
        choices = [int(choice) for choice in choices]
        for idx, _ in enumerate(wo):
            if idx in choices:
                wo[idx] = 0
    return wo


'''
decrease abs() magnitude of random % weights
'''


def decrease_magnitude_readoutweights(wo, alter=False, alter_percentage=0.3,
                                      decrease_percentage=.5):
    if alter:
        choices = random.choices(np.linspace(0, int(len(wo))-1, int(len(wo))),
                                 k=int(alter_percentage*len(wo)))
        choices = [int(choice) for choice in choices]
        for idx, _ in enumerate(wo):
            if idx in choices:
                wo[idx] = wo[idx] * decrease_percentage

    return wo


'''
select random connection, set it and connections around it to 0.
'''


def alter_randomrange_readoutweights(wo, alter=False, how_many_choices=10,
                                     alter_around=10):
    if alter:
        choices = get_choice(how_many_choices, alter_around)
        for idx, _ in enumerate(wo):
            if idx in choices:
                wo[idx] = 0
                # Alter + and - of alter_around
                for alter_idx in np.linspace(idx-alter_around,
                                             idx+alter_around,
                                             2*alter_around+1):
                    if alter_idx < len(wo) and alter_idx > (-1):
                        wo[int(alter_idx)] = 0
    return wo


'''
MODIFICATIONS BELOW EFFECT THE RECURRENT NETWORK CONNECTIONS/NEURONS
'''


'''
Kill random neuron from recurrent network. This is equivalent to killing a
whole row/column of the matrix.
'''


def kill_recurrent_neurons(M, alter=False, how_many_choices=1, alter_around=0):
    if alter:
        non_zero_count = 0
        choices = get_choice(how_many_choices, alter_around)  # Gives choice in 1000
#        print("Selected neurons: {}".format(choices))
        for idx, row in enumerate(M):
            if idx in choices:
                row = np.zeros(np.shape(row))
            M[idx] = row
        for idx, column in enumerate(M.T):
            if idx in choices:
                column = np.zeros(np.shape(column))
            M[idx] = column
    return M


'''
Kill recurrent connections
'''


def kill_recurrent_connections(M, alter=False, how_many_choices=2):
    if alter:
        alter_around = 0
        choices_i = get_choice(how_many_choices, alter_around)  # Gives choice in 1000
        choices_j = get_choice(how_many_choices, alter_around)
#        for c in range(len(choices_i)):
#            print("Selected connections: {}, {}".format(choices_i[c], choices_j[c]))
#        print("For a sparse recurrent network, these connections may already be zero")
        for idx, connection in enumerate(choices_i):
#            print("Connection at {} was {}.".format((connection, choices_j[idx]),
#                                            M[connection, choices_j[idx]]))
            M[connection, choices_j[idx]] = 0
#            print("Connection at {} now is {}.".format((connection, choices_j[idx]),
#                                            M[connection, choices_j[idx]]))
#            print()
    return M


'''
Decrease recurrent connection weight
'''


def decrease_recurrent_connections(M, alter=False, how_many_choices=2,
                                   decrease_mutliplier=.5):
    if alter:
        alter_around = 0
        choices_i = get_choice(how_many_choices, alter_around)  # Gives choice in 1000
        choices_j = get_choice(how_many_choices, alter_around)
#        for c in range(len(choices_i)):
#            print("Selected connections: {}, {}".format(choices_i[c], choices_j[c]))
#        print("For a sparse recurrent network, these connections may already be zero")
        for idx, connection in enumerate(choices_i):
#            print("Connection at {} was {}.".format((connection, choices_j[idx]),
#                                            M[connection, choices_j[idx]]))
            M[connection, choices_j[idx]] = M[connection, choices_j[idx]] * decrease_mutliplier
#            print("Connection at {} now is {}.".format((connection, choices_j[idx]),
#                                            M[connection, choices_j[idx]]))
#            print()
    return M


if __name__ == "__main__":
    # M = np.ones((10, 10))
    # # wo = np.ones((1000, 1))
    # # print(type(len(wo)))
    # M = decrease_recurrent_connections(M, alter=True, how_many_choices=2,
    #                                    decrease_mutliplier=.5)
    # print(M)
    pass
