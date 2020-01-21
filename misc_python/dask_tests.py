
from distributed import Client
from time import sleep
import random

def inc(x):
    sleep(random.random() / 10)
    print ("inc")
    return x + 1

def dec(x):
    sleep(random.random() / 10)
    return x - 1

def add(x, y):
    sleep(random.random() / 10)
    return x + y


if __name__ == '__main__':

    client = Client()

    incs = client.map(inc, range(100))
    decs = client.map(dec, range(100))
    adds = client.map(add, incs, decs)
    total = client.submit(sum, adds)

    del incs, decs, adds
    total.result()

