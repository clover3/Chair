

class Parent:
    def method_A(self):
        print("Parent A")
        self.method_A2()

    def method_A2(self):
        print("Parent A2")
        self.method_A3()

    def method_A3(self):
        print("Parent A3")
        self.method_A4()

    def method_A4(self):
        print("Parent A4")


class Child(Parent):
    def method_A2(self):
        print("Child A2")
        self.method_A3()

    def method_A4(self):
        print("Child A4")


parent = Parent()
parent.method_A()

child = Child()
child.method_A()

