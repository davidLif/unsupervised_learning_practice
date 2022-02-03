import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm  # pip install --upgrade --user hmmlearn
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler





def bic_general(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    bic = np.log(len(X))*k - 2*likelihood_fn(X)
    return bic


def aic_general(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    aic = (- 2 / len(X))*likelihood_fn(X) + (2 * k) / len(X)
    return aic


def bic_hmmlearn(X):
    lowest_bic = np.infty
    lowest_aic = np.infty
    best_n_states_a = 0
    best_n_states_b = 0
    bic = []
    aic = []
    num_params = []
    n_states_range = range(1,25)
    for n_components in n_states_range:
        hmm_curr = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
        hmm_curr.fit(X)

        # Calculate number of free parameters
        # free_parameters = for_means + for_covars + for_transmat + for_startprob
        # for_means & for_covars = n_features*n_components
        n_features = hmm_curr.n_features
        free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)
        num_params.append(free_parameters)
        bic_curr = bic_general(hmm_curr.score, free_parameters, X)
        aic_curr = aic_general(hmm_curr.score, free_parameters, X)
        bic.append(bic_curr)
        aic.append(aic_curr)
        if bic_curr < lowest_bic:
            lowest_bic = bic_curr
            best_n_states_b = n_components
            best_hmm_b = hmm_curr
        if aic_curr < lowest_aic:
            lowest_aic = aic_curr
            best_n_states_a = n_components
            best_hmm_a = hmm_curr

    def plot_bic_vs_nstate():
        plt.plot(list(n_states_range), bic)
        plt.xlabel("n states")
        plt.xticks(list(n_states_range), list(n_states_range))
        plt.ylabel("BIC score")
        plt.title("BIC per Number of Hidden States")
        plt.savefig("bic_per_n_state.png")

    plot_bic_vs_nstate()
    plt.clf()

    def plot_aic_vs_nstate():
        plt.plot(list(n_states_range), aic)
        plt.xlabel("n states")
        plt.xticks(list(n_states_range), list(n_states_range))
        plt.ylabel("AIC score")
        plt.title("AIC per Number of Hidden States")
        plt.savefig("aic_per_n_state.png")

    plot_aic_vs_nstate()

    return best_hmm_a, bic, num_params, best_n_states_a


def fitHMM(Q, n_states=2):
    # fit Gaussian HMM to Q
    # Q = np.reshape(Q, [len(Q), 5])
    # Q = Q[["Open","High","Low","Close"]]
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000).fit(Q)
    # classify each observation as state 0 or 1
    hidden_states = model.predict(Q)
    return hidden_states, model


def plotTimeSeries(Q, hidden_states, ylabel, filename):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = np.arange(len(Q))
    masks = hidden_states == 0
    midPrice =(Q["High"] - Q["Low"])/2.
    ax.scatter(xs[masks], midPrice[masks], c='r', label='State0')
    masks = hidden_states == 1
    ax.scatter(xs[masks], midPrice[masks], c='b', label='State1')
    ax.plot(xs, Q, c='k')

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()

    return None


def visualize_data(df, hidden_states_options, file_name, col_name='principle_cmp1'):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df[col_name]))
    plt.xticks(range(0, df.shape[0], 250), df['Date'].dt.date.loc[::250], rotation=15)
    # plt.xticks(range(0, df.shape[0]), list(df.index), rotation=45)


    colors_array = ["red", "blue", "green", "yellow", "purple", "orange", "black", "grey", "aqua"
        , "silver", "olive", "pink", "darkseagreen", "mediumaquamarine", "indigo", "plum", "orchid", "peru", "coral", "salmon"
                    ,"khaki", "darkred", "wheat", "sienna", "violet"]
    for i in hidden_states_options:
        masks = df['hidden_states'] == i
        plt.scatter(np.arange(len(df))[masks], (df[col_name])[masks], c=colors_array[i]
                    , label='State' + str(i))

    plt.xlabel('Date', fontsize=18)
    plt.ylabel(col_name, fontsize=18)
    plt.savefig(file_name)
    #plt.show()


def main():
    # load stocks data
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us2010-2017.txt'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df["midPrice"] = df["High"] - df["Low"] / 2.
    df["normed_midPrice"] = (df["High"] - df["Low"]) / df["High"]
    X = df[["Open","High", "Low","Close" ]]
    model, bics, num_params, best_n_states = bic_hmmlearn(X)
    n_states = best_n_states
    df["hidden_states"] = model.predict(X)
    visualize_data(df, range(n_states), "{0}_hidden_states.png".format(n_states), col_name="midPrice")


if __name__ == "__main__":
    main()