import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm  # pip install --upgrade --user hmmlearn


def fitHMM(Q, n_states=2, nSamples=0):
    # fit Gaussian HMM to Q
    Q = np.reshape(Q, [len(Q), 1])
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000).fit(Q)
    # classify each observation as state 0 or 1
    hidden_states = model.predict(Q)
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q, [len(Q), 1]))

    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    # #check if the model was converged
    # model.monitor_.converged
    return hidden_states, mus, sigmas, P, logProb, samples


def plotTimeSeries(Q, hidden_states, ylabel, filename):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = np.arange(len(Q)) + 1970
    masks = hidden_states == 0
    ax.scatter(xs[masks], Q[masks], c='r', label='State0')
    masks = hidden_states == 1
    ax.scatter(xs[masks], Q[masks], c='b', label='State1')
    ax.plot(xs, Q, c='k')

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()

    return None


def visualize_data(df):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()


def main():
    # load stocks data
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])

    hidden_states, mus, sigmas, P, logProb, samples = fitHMM(df, nSamples=100)
    plt.switch_backend('agg')  # turn off display when running with Cygwin
    plotTimeSeries(df, hidden_states, 'MidPrice', 'StateTseries_Log.png')
