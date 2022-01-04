import hmmlearn as hmm  # pip install --upgrade --user hmmlearn

X = None
remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
remodel.fit(X)


#check if the model was converged
remodel.monitor_.converged