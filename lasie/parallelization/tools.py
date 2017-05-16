#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:39:14 2017

@author: root
"""

import time

# =========================================================================== #

import pp


def ppmap(function, sequence, deps=(), modules=()):
    job_server = pp.Server()    
    jobs = tuple()
    for arg in sequence:
        jobs += job_server.submit(function, tuple([arg, ]), deps, modules), 
    return [job() for job in jobs]


# =========================================================================== #

import multiprocessing as multi

def  multimap(function, sequence):
    pool = multi.Pool()
    return pool.map(function, sequence)

import multiprocessing


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

# =========================================================================== #

if __name__ == "__main__":
    
    def myfunc(N):
        "A simple function"
        res = 1
        for i in range(N):
            res *= 1+i
        return res

#    t0 = time.time()    
#    respp = ppmap(myfunc, range(10000, 10050))
#    tpp = time.time() - t0
#    print('tpp={}'.format(tpp))

    t0 = time.time()    
    resmulti = multimap(myfunc, range(10000, 10050))
    tmulti = time.time() - t0
    print('tmulti={}'.format(tmulti))
                      
    t0 = time.time()    
    resmulti = parmap(myfunc, range(10000, 10050))
    tparmap = time.time() - t0
    print('tparmap={}'.format(tparmap))
                      
    t0 = time.time()  
    resmap = map(myfunc, range(10000, 10050))
    tmap = time.time() - t0
    print('tmap={}, '.format(tmap))
    
    
            

