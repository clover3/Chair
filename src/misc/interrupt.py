import time


def sleeper():
    time.sleep(100)


if __name__ == "__main__":
    try:
        sleeper()
    except Exception as e:
        print("Catched exeption")
        print(e)
        raise
    except KeyboardInterrupt as e :
        print("Caught KeyboardInterrup")
        print(e)
        raise