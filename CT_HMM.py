import numpy as np
from scipy import linalg as la
from scipy.stats import gamma
from scipy.special import digamma, polygamma


class CT_HMM():
    def __init__(self, data, time_intervals, transition_matrix, start_prob, g_shape, g_scale):
        self.sequence = data
        self.time_intervals = time_intervals
        self.q = np.array(transition_matrix)
        self.n_states = np.shape(self.q)[0]
        self.gamma_shape = g_shape
        self.gamma_scale = g_scale
        self.init_prob = self.format_init_probs(start_prob)
        self.emiss_probs = self.format_emiss_probs(data)
        self.trans_probs_given_time = self.format_trans_probs_gt()
        self.trans_probs_all_time = self.format_trans_probs_anyt()

    ########################################################################################################################
    # helper functions
    def format_trans_probs_gt(self):
        trans_probs_gt = dict()
        for time in self.time_intervals.values():
            for t in range(len(time)):
                if time[t]:
                    trans_probs_gt[time[t]] = la.expm(self.q * float(time[t]))
        return trans_probs_gt

    def format_trans_probs_anyt(self):
        trans_prob_anyt = dict()
        for time in self.time_intervals.values():
            for t in range(len(time)):
                if time[t] is not None:
                    trans_prob_anyt[time[t]] = dict()
                    for state in range(self.n_states):
                        trans_prob_anyt[time[t]][state] = dict()
        for item in trans_prob_anyt:
            trans_prob_anyt[item] = np.zeros((self.n_states, self.n_states))
        return trans_prob_anyt

    def format_init_probs(self, s_probs):
        init_probs = dict()
        for i in range(len(s_probs)):
            init_probs[i] = s_probs[i]
        return init_probs

    def format_emiss_probs(self, data):
        # create a dictionary with the emission probabilities
        emiss_probs = dict()
        for state in range(self.n_states):
            emiss_probs[state] = dict()

        # apply the gamma distribution to the emission probabilities
        for sequence in data:
            for i in range(len(data[sequence])):
                for state in range(self.n_states):
                    emiss_probs[state][(data[sequence][i])] = gamma.pdf(float(data[sequence][i]),
                                                                        a=self.gamma_shape[state],
                                                                        scale=self.gamma_scale[state])
                    if emiss_probs[state][(data[sequence][i])] == 0:
                        emiss_probs[state][(data[sequence][i])] = 1e-300
        return emiss_probs

    def format_exp_time(self):
        et = dict()
        init_1 = np.zeros((self.n_states, self.n_states))
        init_2 = np.zeros((self.n_states, self.n_states, self.n_states))
        for i in range(self.n_states):
            et[i] = 0

        format_exmp = dict()
        for j in self.trans_probs_all_time:
            m = [0] * self.n_states
            m = np.array(m)
            m = m.reshape(1, self.n_states)
            for k in range(self.n_states):
                temp = np.array(self.trans_probs_all_time[j][k])
                temp = temp.reshape(1, self.n_states)
                m = np.append(m, temp, axis=0)
            m = np.delete(m, list(range(0, self.n_states)))
            m = m.reshape(self.n_states, self.n_states)
            format_exmp[j] = m
        for i in range(self.n_states):
            for time in self.trans_probs_all_time:
                m = np.zeros((self.n_states, self.n_states))
                m[i][i] = 1
                n = np.bmat(([self.q, m], [init_1, self.q]))
                init_2[i] = ((la.expm(n * float(time)))[0:self.n_states, self.n_states:2 * self.n_states]) / (
                self.trans_probs_given_time[time])
                et[i] += (format_exmp[time] * init_2[i]).sum().sum()
        return et

    def format_exp_n_transitions(self):
        ent = dict()
        init = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            ent[i] = dict()
            for j in range(self.n_states):
                ent[i][j] = 0

        format_exmp = dict()
        for j in self.trans_probs_all_time:
            m = [0] * self.n_states
            m = np.array(m)
            m = m.reshape(1, self.n_states)
            for k in range(self.n_states):
                temp = np.array(self.trans_probs_all_time[j][k])
                temp = temp.reshape(1, self.n_states)
                m = np.append(m, temp, axis=0)
            m = np.delete(m, list(range(0, self.n_states)))
            m = m.reshape(self.n_states, self.n_states)
            format_exmp[j] = m

        for i in range(self.n_states):
            for j in range(self.n_states):
                for time in self.trans_probs_all_time:
                    m = np.zeros((self.n_states, self.n_states))
                    m[i][j] = 1
                    n = np.bmat([[self.q, m], [init, self.q]])
                    l = self.q[i][j] * (la.expm(n * float(time))[0:self.n_states, self.n_states:2 * self.n_states]) / (
                    self.trans_probs_given_time[time])
                    ent[i][j] += (format_exmp[time] * l).sum().sum()
        return ent

    # newtonian_step and gamma_MLE are used to calulate the new shape and scale of the state emissions
    def newtonian_step(self, param, xbar, logxbar):
        parameter = param
        if parameter <= 0:
            parameter = 1e-20
        if (1 / parameter - polygamma(1, parameter)) == 0:
            updated_params = 0
        else:
            updated_params = (np.log(parameter) - digamma(parameter) - np.log(xbar) + logxbar) / (
                        1 / parameter - polygamma(1, parameter))
        update = parameter - updated_params
        return update

    def gamma_MLE(self, xbar, logxbar, sequence):
        CONVERGENCE_METRIC = 0.0001
        denom = 0
        obsv_counter = 0
        for obsv in sequence.values():
            for value in obsv:
                denom += (float(value) - xbar) ** 2
                obsv_counter += 1
        moment_estimator = (obsv_counter * xbar ** 2) / (denom)
        mom = [moment_estimator]
        mom.append(self.newtonian_step(mom[0], xbar, logxbar))
        iteration = 1
        while (abs(mom[iteration] - mom[iteration - 1]) > CONVERGENCE_METRIC):
            if (iteration > 10_000): raise ValueError("Max Iterations Exceeded")
            mom.append(self.newtonian_step(mom[iteration], xbar, logxbar))
            iteration += 1
        theta = xbar / mom[iteration - 1]
        return mom[-1], theta

    ##################################################################################################################
    # functions for CT-HMM
    def forward(self):
        fwd = dict()
        for t in self.time_intervals:
            fwd[t] = dict()
            for state in range(self.n_states):
                fwd[t][state] = [0] * len(self.sequence[t])
                fwd[t][state][0] = self.init_prob[state] * self.emiss_probs[state][self.sequence[t][0]]

            obsv_counter = 1
            for time in self.time_intervals[t]:
                if time is not None:
                    for state in range(self.n_states):
                        sum_transition_probs = 0
                        for i in range(self.n_states):
                            trans_prob = fwd[t][i][obsv_counter - 1] * self.trans_probs_given_time[time][i][state]
                            sum_transition_probs += trans_prob
                        fwd[t][state][obsv_counter] = (self.emiss_probs[state][self.sequence[t][obsv_counter]]
                                                       * sum_transition_probs)
                    obsv_counter += 1
        return fwd

    def backward(self):
        bwd = dict()
        for t in self.time_intervals:
            bwd[t] = dict()
            for state in range(self.n_states):
                bwd[t][state] = [0] * len(self.sequence[t])
                bwd[t][state][len(self.sequence[t]) - 1] = 1

            obsv_counter = len(self.sequence[t]) - 2
            for time in reversed(self.time_intervals[t]):
                if time is not None:
                    for state in range(self.n_states):
                        sum_transition_probs = 0
                        for i in range(self.n_states):
                            sum_transition_probs += (self.trans_probs_given_time[time][state][i]
                                                     * self.emiss_probs[i][self.sequence[t][obsv_counter + 1]]
                                                     * bwd[t][i][obsv_counter + 1])
                            bwd[t][state][obsv_counter] = sum_transition_probs
                obsv_counter -= 1
        return bwd

    def exp_max(self, post_prob):
        et = self.format_exp_time()
        ent = self.format_exp_n_transitions()

        # update the transition matrix
        for i in range(self.n_states):
            total = 0
            for j in range(self.n_states):
                if (i != j):
                    self.q[i][j] = ent[i][j] / et[j]
                    total += self.q[i][j]
            self.q[i][i] = total * -1

        assert (not [row for row in range(len(self.q)) if round(sum(self.q[row]), 3) != 0])

        # we are updating the gamma shape and scale
        xbar = [0] * self.n_states
        logxbar = [0] * self.n_states
        sum_numx = [0] * self.n_states
        sum_denom = [0] * self.n_states
        sum_num_logx = [0] * self.n_states

        for sequence in self.sequence:
            for i in range(len(self.sequence[sequence])):
                for j in range(self.n_states):
                    sum_numx[j] += post_prob[sequence][j][i] * float(self.sequence[sequence][i])
                    sum_denom[j] += post_prob[sequence][j][i]
                    sum_num_logx[j] += post_prob[sequence][j][i] * np.log(float(self.sequence[sequence][i]))
        new_init_probs = np.zeros((self.n_states, 1))
        for i in range(self.n_states):
            for sequence in self.sequence:
                new_init_probs[i] += post_prob[sequence][i][1]
        init_probs_denom = sum(new_init_probs)
        for i in range(self.n_states):
            self.init_prob[i] = new_init_probs[i] / init_probs_denom
        for i in range(self.n_states):
            xbar[i] = sum_numx[i] / sum_denom[i]
            logxbar[i] = sum_num_logx[i] / sum_denom[i]
            self.gamma_shape[i], self.gamma_scale[i] = self.gamma_MLE(xbar[i], logxbar[i], self.sequence)
        return self.q

    def likelihood(self, f, b):
        # find the likelihood of the fwd/bwd results
        sequence_likelihood = dict()

        temp_trans_matrix = dict()
        for _ in self.time_intervals.values():
            for time in range(len(_)):
                if _[time] is not None:
                    temp_trans_matrix[_[time]] = dict()

        for t in self.time_intervals:
            sequence_likelihood[t] = 0
            likelihood = 0
            for state in range(self.n_states):
                likelihood += f[t][state][-1]
            if (likelihood <= 0):
                likelihood = 1e-10
            sequence_likelihood[t] = np.log10(likelihood)

            # we also have to update the probabity of transition for all time
            counter = 0
            for interval in self.time_intervals[t]:
                if interval:
                    denom = 0
                    temp_trans_matrix[interval] = np.zeros((self.n_states, self.n_states))
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            temp_trans_prob = (f[t][i][counter] * self.trans_probs_given_time[interval][i][j]
                                               * self.emiss_probs[j][self.sequence[t][counter + 1]]
                                               * b[t][j][counter + 1])
                            temp_trans_matrix[interval][i][j] += temp_trans_prob
                            denom += temp_trans_prob
                    for k in range(self.n_states):
                        for l in range(self.n_states):
                            if (denom == 0):
                                denom = 1e-10
                            temp_trans_matrix[interval][k][l] = temp_trans_matrix[interval][k][l] / denom
                            self.trans_probs_all_time[interval][k][l] += temp_trans_matrix[interval][k][l]
                    counter += 1

        log_likelihood = 0
        for ll in sequence_likelihood:
            log_likelihood += sequence_likelihood[ll]

        return log_likelihood

    # finding the posterior prob for the expectation-maximization algo
    def posterior_probability(self, f, b):
        post = dict()
        obsv_likelihood = dict()
        for t in self.time_intervals:
            post[t] = dict()
            obsv_likelihood[t] = [0] * len(self.sequence[t])

            for i in range(len(self.sequence[t])):
                total = 0
                for j in range(self.n_states):
                    total += (f[t][j][i] * b[t][j][i])
                obsv_likelihood[t][i] = total

            for i in range(self.n_states):
                post[t][i] = [0] * len(self.sequence[t])
                for j in range(len(self.sequence[t])):
                    if (obsv_likelihood[t][j] == 0):
                        post[t][i][j] = 1e-10
                    else:
                        post[t][i][j] = (f[t][i][j] * b[t][i][j]) / obsv_likelihood[t][j]
        return post

    ##################################################################################################################
    def train_model(self):
        CONVERGENCE_METRIC = 0.0001
        log_likelihood = list()

        fwd = self.forward()
        bwd = self.backward()
        pp = self.posterior_probability(fwd, bwd)
        final_loglikelihood = self.likelihood(fwd, bwd)
        update_parameters = self.exp_max(pp)
        log_likelihood.append(final_loglikelihood)

        fwd = self.forward()
        bwd = self.backward()
        pp = self.posterior_probability(fwd, bwd)
        final_loglikelihood = self.likelihood(fwd, bwd)
        update_parameters = self.exp_max(pp)
        log_likelihood.append(final_loglikelihood)

        i = 1
        while (abs(log_likelihood[i] - log_likelihood[i - 1]) > CONVERGENCE_METRIC):
            fwd = self.forward()
            bwd = self.backward()
            pp = self.posterior_probability(fwd, bwd)
            final_loglikelihood = self.likelihood(fwd, bwd)
            update_parameters = self.exp_max(pp)
            log_likelihood.append(final_loglikelihood)
            i = i + 1

        final_num_parameters = 2 * len(self.gamma_shape) + (len(self.gamma_shape) * len(self.gamma_shape)) - 1
        aic_score = (2 * final_num_parameters) - (2 * log_likelihood[-1])
        num_observations = sum(len(x) for x in self.sequence.values())
        bic_score = (final_num_parameters * np.log(num_observations)) - (2 * log_likelihood[-1])
        return (self.init_prob, update_parameters, log_likelihood,
                aic_score, bic_score,
                self.gamma_shape, self.gamma_scale)
