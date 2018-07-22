

def calc_next_params(domain, trials):
    next_params = {}
    random_params = domain.random()
    for index, fieldname in enumerate(domain.fieldnames):
        next_params[fieldname] = random_params[index]
    return [next_params]
