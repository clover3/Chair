import tensorflow_probability as tfp


def prac():
    prob = [0.5]
    geo = tfp.distributions.Geometric(prob)
    print(geo.sample(30))




prac()
