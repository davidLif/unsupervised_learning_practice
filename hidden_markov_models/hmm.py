import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm  # pip install --upgrade --user hmmlearn


def fitHMM(Q, n_states=2):
    # fit Gaussian HMM to Q
    Q = np.reshape(Q, [len(Q), 5])
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000).fit(Q)
    # classify each observation as state 0 or 1
    hidden_states = model.predict(Q)
    return hidden_states


def plotTimeSeries(Q, hidden_states, ylabel, filename):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = np.arange(len(Q))
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


def visualize_data(df, hidden_states_options, file_name):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date_delta'].loc[::500], rotation=45)

    colors_array = ["red", "blue", "green", "yellow", "purple", "orange", "black", "grey", "aqua", "silver", "olive"]
    for i in hidden_states_options:
        masks = df['hidden_states'] == i
        plt.scatter(np.arange(len(df))[masks], ((df['Low'] + df['High']) / 2.0)[masks], c=colors_array[i]
                    , label='State' + str(i))

    plt.xlabel('Date_delta', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.savefig(file_name)
    #plt.show()


def main():
    # load stocks data
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_delta'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
    df = df.drop(columns=['Date'])
    df['hidden_states'] = 1

    for n_states in [2, 4, 6, 8]:
        df = df.drop(columns=['hidden_states'])
        hidden_states = fitHMM(df, n_states)

        df['hidden_states'] = hidden_states
        #plt.switch_backend('agg')  # turn off display when running with Cygwin
        samples = df.sample(200)
        visualize_data(samples, range(n_states), "graph_for_{0}_hidden_states.svg".format(n_states))


if __name__ == "__main__":
    main()