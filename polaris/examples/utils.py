from polaris import STATUS_SUCCESS


def pseudo_train(params, exp_info):
    lr_squared = ((params['lr'] - 0.006) ** 2) * 1e+5
    weight_decay_squared = ((params['weight_decay'] - 0.02) ** 2) * 1e+5
    loss = lr_squared + weight_decay_squared
    return {
        'loss':  loss,
        'status': STATUS_SUCCESS,
    }
