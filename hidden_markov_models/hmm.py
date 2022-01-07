import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm  # pip install --upgrade --user hmmlearn
from sklearn.manifold import MDS


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
    plt.plot(range(df.shape[0]), (df['principle_cmp1']))
    plt.xticks(range(0, df.shape[0], 500), df['Date_delta'].loc[::500], rotation=45)

    colors_array = ["red", "blue", "green", "yellow", "purple", "orange", "black", "grey", "aqua"
        , "silver", "olive", "pink", "darkseagreen"]
    for i in hidden_states_options:
        masks = df['hidden_states'] == i
        plt.scatter(np.arange(len(df))[masks], (df['principle_cmp1'])[masks], c=colors_array[i]
                    , label='State' + str(i))

    plt.xlabel('Date_delta', fontsize=18)
    plt.ylabel('principle_cmp1', fontsize=18)
    plt.savefig(file_name)
    #plt.show()


def main():
    # load stocks data
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_delta'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
    df = df.drop(columns=['Date'])

    # Dim reduction to
    df = df.sample(3000)  # My computer doesn't have enough RAM for more than that.
    model = MDS(n_components=1)
    transformed_data = model.fit_transform(df[['Open', 'High', 'Low', 'Close']])
    transformed_data_df = pd.DataFrame(data=transformed_data
                                       , columns=['principle_cmp1'])
    transformed_data_df['Date_delta'] = df['Date_delta']
    transformed_data_df['hidden_states'] = 1

    for n_states in [13]:
        transformed_data_df.drop(columns=['hidden_states'])
        hidden_states = fitHMM(df, n_states)

        transformed_data_df['hidden_states'] = hidden_states

        #plt.switch_backend('agg')  # turn off display when running with Cygwin
        samples = transformed_data_df.sample(200)
        visualize_data(samples, range(n_states), "graph_with_dm_{0}_hidden_states.svg".format(n_states))


if __name__ == "__main__":
    main()