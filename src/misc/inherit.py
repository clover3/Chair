


class Mother:

    def get_number(self):
        return 10
    def get_output(self):
        print(self.get_number())


class Child(Mother):

    def get_number(self):
        return 5


if __name__ == "__main__":
    c = Child()
    c.get_output()