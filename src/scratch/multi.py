import multiprocessing as mp
import time
def foo(q, i):
    time.sleep(10)
    q.put('hello'+str(i))

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    processes = []
    for i in list(range(10)):
        p = mp.Process(target=foo, args=(q, i))
    p.start()
    print("ji")
    print(q.get())
    p.join()