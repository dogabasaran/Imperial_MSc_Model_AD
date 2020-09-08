import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.stats as stats
from tqdm import tqdm
from objective_functions import *
import random
import sys
import os

'''
Uses architecture C from Sussillo and Abbott, with RLS. all-2-all connectivity allows large optimization. We can maintain
a single inverse correlation matrix for the network. There is no longer a feedback loop from the output unit.
'''

def architectureC():
    # Set up network parameters
    N = 1000
    print("Recurrent network has {} neurons".format(N))
    p = 1.0
    g = 1.5        # g greater than 1 leads to chaoric networks
    alpha = 1.0
    nsecs = 1440
    dt = .1  #.1
    learn_every = 5
    scale = 1.0/math.sqrt(p*N)

    rvs = stats.norm(loc=0, scale=1).rvs
    M = sparse.random(m=N, n=N, density=p, data_rvs=rvs)*g*scale
    M = M.todense()

    # M = np.random.normal(loc=0, scale=1, size=(N,N))

    nRec2Out = N
    wo = np.zeros((nRec2Out, 1))
    dw = np.zeros((nRec2Out, 1))

    simtime = np.linspace(start=0, stop=int(nsecs-dt), num=int(nsecs/dt))
    simtime_len = len(simtime)
    simtime2 = np.linspace(start=int(nsecs), stop=int(2*nsecs), num=int(nsecs/dt))
    '''
    periodic
    '''
    # Make sine waves
    # Amplitude is lower because wf is implied as all ones, which is half the strength of wf.
    amp = 0.7
    freq = 1/60
    ft = (amp/1.0)*np.sin(1.0*(math.pi)*freq*simtime) + \
         (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime) + \
         (amp/6.0)*np.sin(3.0*(math.pi)*freq*simtime) + \
         (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime)
    ft = ft/1.5

    ft2 = (amp/1.0)*np.sin(1.0*(math.pi)*freq*simtime2) + \
          (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime2) + \
          (amp/6.0)*np.sin(3.0*(math.pi)*freq*simtime2) + \
          (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime2)
    ft2 = ft2/1.5
    '''
    discontinuous
    '''
    # noise = np.random.normal(0,.08,14400)
    #
    # amplitude = 4
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
    # ft = ft+noise
    #
    '''
    aperiodic
    '''
    # amp = 0.7
    # freq = 1/60
    # ft = (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime) #+ \
    #      # (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime)
    # ft = ft/1.5
    # funcsinc = np.sinc(simtime/80 - 10) - np.sinc(simtime/80 - 7) + np.sinc(simtime/70 - 10)\
    # + np.sinc(simtime/70 - 5) - np.sinc(simtime/80 -13  ) - np.sinc(simtime/80) + np.sinc(simtime/80 - 60)
    # ft = .8* (funcsinc + ft)
    # ft2 = ft

    wo_len = np.zeros(simtime_len)
    zt = np.zeros(simtime_len)
    zpt = np.zeros(simtime_len)

    x0 = 0.5*np.random.normal(loc=0, scale=1, size=(N, 1))
    z0 = 0.5*np.random.normal(loc=0, scale=1, size=(1, 1))

    x = x0
    r = np.tanh(x)
    z = z0
    # xp = x0

    ti = 0
    P = (1.0/alpha) * np.eye(nRec2Out)
    print("Training...")
    for t in tqdm(simtime):
        # sim, so x(t) and r(t) are created
        x = (1.0-dt)*x + M*(r*dt) #+ wf*(z*dt) No more feedback coming from the readout unit.
        r = np.tanh(x)
        z = np.transpose(wo)*r

        if ti % learn_every == 0:
            # Update inverse correlation matrix
            k = P*r
            rPr = np.transpose(r) * k
            c = 1.0/(1.0 + rPr)
            P = P - k*(np.transpose(k*c))

            # Update the error for the linear readouts
            e = z - ft[ti]
            # Update the output weights
            dw = -k * c * e
            wo = wo + dw

            # update the internal weight matrix using the outpu's erroR
            M = M + np.ones((N, 1))*np.transpose(dw)
        # Store the output of the system
        zt[int(ti)] = z
        wo_len[int(ti)] = np.sqrt(np.transpose(wo)*wo)
        # Increment time step
        ti += 1

    # Calculate Mean absolute error
    train_error_avg = np.sum(np.absolute(zt-ft))/simtime_len
    print("Mean Absolute Error during training: {}.".format(np.round(train_error_avg, 5)))
    # Plot training performance
    # fig, axs = plt.subplots(2, figsize=(14, 7))
    # fig.suptitle('Training')
    # plt.subplots_adjust(hspace=0.3)
    # axs[0].set_ylabel("f and z")
    # axs[0].set_xlabel("Time")
    # axs[0].plot(simtime, ft, color='r', label="f")
    # axs[0].plot(simtime, zt, color='b', label="z")
    # axs[0].legend()
    # axs[1].set_ylabel("|w|")
    # axs[1].set_xlabel("Time")
    # axs[1].plot(simtime, wo_len, 'r', label='outputweights')
    # fig.savefig("training.png", bbox_inches='tight')
    params = N, g, alpha, nsecs, dt, learn_every, scale, simtime, simtime2, simtime_len, ft, ft2, zt, zpt, x, r
    model = M, wo, params

    return model, train_error_avg


def alter_architectureC(model, figures_dir, run, alter_opt):
    M, wo, params = model
    N, g, alpha, nsecs, dt, learn_every, scale, simtime, simtime2, simtime_len, ft, ft2, zt, zpt, x, r = params

    # Altering synapses
    # if alter_opt == 0:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=100,  # 1%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 1:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=200, #  2%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 2:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=300, # 3%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 3:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=400,  # 4%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 4:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=500, # 5%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 5:
    #     M = decrease_recurrent_connections(M, alter=True, how_many_choices=600, # 6%
    #                                        decrease_mutliplier=.5)
    # elif alter_opt == 6:
    #     M = kill_recurrent_connections(M, alter=True, how_many_choices=100)     # kills 100/(5k*5k)
    # elif alter_opt == 7:
    #     M = kill_recurrent_connections(M, alter=True, how_many_choices=200)     # kills 200/(5k*5k)
    # elif alter_opt == 8:
    #     M = kill_recurrent_connections(M, alter=True, how_many_choices=300)     # kills 300/(5k*5k)
    # # Killing neurons
    # elif alter_opt == 9:
    #     M = kill_recurrent_neurons(M, alter=True, how_many_choices=1, alter_around=0)
    # elif alter_opt == 10:
    #     M = kill_recurrent_neurons(M, alter=True, how_many_choices=2, alter_around=0)
    # elif alter_opt == 11:
    #     M = kill_recurrent_neurons(M, alter=True, how_many_choices=3, alter_around=0)
    # elif alter_opt == 12:
    #     M = kill_recurrent_neurons(M, alter=True, how_many_choices=4, alter_around=0)
    # elif alter_opt == 9:
    #     print("No alteration made.")
    # else:
    #     print("Exiting, no option like that...")
    #     sys.exit()
    # for i in range(5):
        # print("excitability: {}".format(-1.999 + i/1000))

    ti = 0
    for t in simtime:
        # sim, so x(t) and r(t) are created
        # Not updating the weights wo, because we are no longer training

        # M = kill_recurrent_neurons(M, alter=True, how_many_choices=1, alter_around=0)

        x = (1.0 - dt)*x + M*(r*dt)  # no more + wf*(z*dt)
        r = np.tanh(x)

        # For decrease in excitability, do below:
        #r = (2/(1 + np.exp(-(1.996)*x))-1)

        z = np.transpose(wo)*r
        zpt[ti] = z
        ti += 1
    test_error_avg = np.sum(np.absolute(zpt-ft))/simtime_len
    print("Mean Absolute Error during testing: {}.".format(np.round(test_error_avg, 5)))

    # Plot testing performance
    fig, axs = plt.subplots(2, figsize=(14, 7))
    # fig.suptitle('Testing')
    plt.subplots_adjust(hspace=0.3)
    axs[0].set_ylabel("f and z")
    axs[0].set_xlabel("Time")
    axs[0].set_title("Unaltered function traceback")
    axs[0].plot(simtime, ft, color='r', label="Objective")
    axs[0].plot(simtime, zt, color='b', label="Test")
    axs[0].legend()

    axs[1].set_ylabel("f and z")
    axs[1].set_xlabel("Time")
    axs[1].set_title("Altered function traceback")
    axs[1].plot(simtime, ft2, 'r', label='outputweights')
    axs[1].plot(simtime, zpt, 'b', label='outputweights')

    # Display plots
    # fig.savefig(os.path.join(figures_dir, str(run)+"7.pdf"), bbox_inches='tight')
    plt.close()
    plt.show()
    return test_error_avg


if __name__ == "__main__":
    # pass
    model, train_error = architectureC()
    figdir = "final_figs"
    if not os.path.exists(figdir):
       os.makedirs(figdir)
    alter_architectureC(model, figdir, 0,0)
