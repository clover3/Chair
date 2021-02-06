

class StemmedToken:
    def __init__(self, raw_token, stemmed_token):
        self.raw_token = raw_token
        self.stemmed_token = stemmed_token

    def __str__(self):
        return self.stemmed_token

    def get_raw_token(self):
        return self.raw_token

    def get_stemmed_token(self):
        return self.stemmed_token