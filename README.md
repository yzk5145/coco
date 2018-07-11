# Count-of-counts Histogram

The code implements the Hc and Hg method used in differentially private count-of-counts histogram.

## Prerequisites

This project is written by Python3. It requires libraries:
1. [Gurobi optimizer](http://www.gurobi.com/products/gurobi-optimizer?utm_source=Google&utm_medium=CPC&utm_term=gurobi&utm_campaign=Brand_N.America_search&campaignid=193283256&adgroupid=8992997136&creative=203314797799&keyword=gurobi&matchtype=e&gclid=CjwKCAjwspHaBRBFEiwA0eM3kYAR8s89M68dGjwnNsTGiPGNmNpkdQq38XhMnK-jqaNGgHvuZNNLLhoCuO0QAvD_BwE)
2. scikit-learn

## Project Layout
```
.
├── algs.py  --------------------  functions of running Hc and Hg method 
├── combine.py  -----------------  generates noisy histogram
├── const.py  
├── data/
├── data.py
├── evaluate.py
├── hhs_utils.py
├── isotonic_reg.py  ------------  optimization by Gurobi solver
├── main.py
├── mechanisms.py
└── PAV.py  ---------------------- optimization by Pool Adjacent Violators Alg
```

## Running the Program



### Input Arguments

input arguments can be found by `python3 main.py -h`

```
$ python3 main.py -h
usage: main.py [-h] [-m {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1}] [-b BUDGET]
               [-d {taxi_lv1,haw_lv1}]

optional arguments:
  -h, --help            show this help message and exit
  -m {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1}, --mode {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1}
                        default is 'isoHc-l1': use Hc method 
                        'isoHc-l1': Hc method with l1 postprocessing
                        'isoHc-l2': Hc method with l2 postprocessing
                        'isoHg-l2': Hg method with l2 postprocessing 
                        'isoHg-l1': Hg method with l1 postprocessing 
  -b BUDGET, --budget BUDGET
                        default is 1.0
  -d {taxi_lv1,haw_lv1}, --dataset {taxi_lv1,haw_lv1}
                        default is taxi dataset at top level

```

### Examlpes

To evaluate the Hc method (L1-isotonic regression) with budget 1.0 on taxi dataset:

```
$ python3 main.py -m isoHc-l1 -b 1.0 -d taxi_lv1
error:  2801.0
```

### To Use Your Data
Use *estimate_hc()* or *estimate_hg_l1()* or *estimate_hg_l2()* in `algs.py`.

For example:
```python
x_hat = estimate_hc(x, seed=20, cum_round=True, iso_norm='iso_l1', budget=budget)
```

## Versioning

* v0.1.0: features including non-hierarchical version of Hc and Hg methods

## Contributors

* **Yu-Hsuan Kuo**: <yzk5145@cse.psu.edu> 