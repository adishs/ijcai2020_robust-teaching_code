IJCAI 2020 -- Understanding the Power and Limitations of Teaching with Imperfect Knowledge
## Prerequisites:
```
Python3
Matplotlib
Numpy
Itertools
Yaml
```

## Running the code
To get results, you will need to run the following scripts:

### For $\Delta_{Q_0}$-imperfect teacher and for $\Delta_{\eta}$-imperfect teacher (Figure 2: a, b, e, f)
```
python teaching_noise.py
```

### For $\Delta_{\Examples}$-imperfect teacher and for $\Delta_{\phi}$-imperfect teacher (Figure 2: c, g, d, h)
```
python teaching_subset.py
```

### Results
After running the above scripts, new plots will be created in output/ directory.

In the __main__ function, the variable number_of_iterations denotes the number of runs used to average the results. Set a smaller number for faster execution.
