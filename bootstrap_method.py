from sklearn.utils import resample
import random
import CT_HMM

def bootstrap(data, intervals):
    iterations = 1_000
    sample_size = int(len(data))

    # for transition probabilities
    trans_00 = list()
    trans_01 = list()
    trans_02 = list()
    trans_10 = list()
    trans_11 = list()
    trans_12 = list()
    trans_20 = list()
    trans_21 = list()
    trans_22 = list()

    # for emission probabilities
    shape_1 = list()
    shape_2 = list()
    shape_3 = list()
    scale_1 = list()
    scale_2 = list()
    scale_3 = list()
    mean_1 = list()
    mean_2 = list()
    mean_3 = list()

    # other
    aic_scores = list()
    bic_scores = list()
    init_prob_1 = list()
    init_prob_2 = list()
    init_prob_3 = list()

    for i in range(iterations):
        sample = resample(list(data.keys()), n_samples=sample_size)
        sequence = dict()
        time = dict()
        for key in sample:
            # allowing for duplicated samples
            if key in sequence.keys():
                new_key = key + str(random.randrange(int(len(data) * 2)))
                sequence[new_key] = data[key]
                time[new_key] = intervals[key]
            else:
                sequence[key] = data[key]
                time[key] = intervals[key]
        estimate_trans_mat = [[-0.892, 0.729, 0.163],
                              [0.814, -1.016, 0.202],
                              [0.163, 0.276, -0.439]]
        estimate_init_probs = [0.6234, 0.2109, 0.1657]
        estimate_gamma_shape = [19.7254, 18.3060, 15.4033]
        estimate_gamma_scale = [0.3153, 0.2158, 0.2184]
        model = CT_HMM.CT_HMM(sequence, time, estimate_trans_mat, estimate_init_probs, estimate_gamma_shape,
                       estimate_gamma_scale)
        init, trans_matrix, log_like, score_1, score_2, g_shape, g_scale = model.train_model()

        # fill in our lists
        init_prob_1.append(init[0])
        init_prob_2.append(init[1])
        init_prob_3.append(init[2])

        trans_00.append(trans_matrix[0][0])
        trans_01.append(trans_matrix[0][1])
        trans_02.append(trans_matrix[0][2])

        trans_10.append(trans_matrix[1][0])
        trans_11.append(trans_matrix[1][1])
        trans_12.append(trans_matrix[1][2])

        trans_20.append(trans_matrix[2][0])
        trans_21.append(trans_matrix[2][1])
        trans_22.append(trans_matrix[2][2])

        shape_1.append(g_shape[0])
        shape_2.append(g_shape[1])
        shape_3.append(g_shape[2])

        scale_1.append(g_scale[0])
        scale_2.append(g_scale[1])
        scale_3.append(g_scale[2])

        mean_1.append(g_shape[0] * g_scale[0])
        mean_2.append(g_shape[1] * g_scale[1])
        mean_3.append(g_shape[2] * g_scale[2])

        aic_scores.append(score_1)
        bic_scores.append(score_2)

    trans_matrix = [trans_00, trans_01, trans_02,
                    trans_10, trans_11, trans_12,
                    trans_20, trans_21, trans_22]
    shapes = [shape_1, shape_2, shape_3]
    scales = [scale_1, scale_2, scale_3]
    mean = [mean_1, mean_2, mean_3]
    init = [init_prob_1, init_prob_2, init_prob_3]
    outcome = [trans_matrix, shapes, scales, mean, init, aic_scores, bic_scores]
    return outcome

