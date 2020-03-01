#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:54:43 2020

@author: joshowen121
"""

def foo(**kwargs):
    for item in kwargs:
        print(item)


foo(b=1, a=2, c=3)

