

def calc_next_params(domain, trials, min_ei):
    next_params = {}
    for index, fieldname in enumerate(domain.fieldnames):
        next_params[fieldname] = domain.random()[index]
    return next_params
