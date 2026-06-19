from kan import KAN


def create_model():

    model = KAN(
        width=[5,16,1],
        grid=5,
        k=3
    )

    return model