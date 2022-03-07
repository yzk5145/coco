# Count-of-counts Histogram

The code implements the non-hierarchical and hierarchical version of Hc and Hg method used in differentially private count-of-counts histogram. 

## Prerequisites

This project is written by Python3. It requires libraries:
1. [Gurobi optimizer](http://www.gurobi.com/products/gurobi-optimizer?utm_source=Google&utm_medium=CPC&utm_term=gurobi&utm_campaign=Brand_N.America_search&campaignid=193283256&adgroupid=8992997136&creative=203314797799&keyword=gurobi&matchtype=e&gclid=CjwKCAjwspHaBRBFEiwA0eM3kYAR8s89M68dGjwnNsTGiPGNmNpkdQq38XhMnK-jqaNGgHvuZNNLLhoCuO0QAvD_BwE)
2. scikit-learn

## Project Layout
```
.
├── algs.py  --------------------  functions of running Hc and Hg method 
├── combine.py  -----------------  generates noisy histogram
├── consistency.py  -----------------  matching and variance estimate
├── const.py  
├── data/
├── data.py
├── evaluate.py
├── hhs_utils.py
├── hierarchy.py  ----------------- generate consistency estimates across hierarchy
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
usage: main.py [-h]
               [-m {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1,HcHcHc-3lv,HgHgHg-3lv}]
               [-b BUDGET] [-d {taxi_lv1,haw_lv1}]

optional arguments:
  -h, --help            show this help message and exit
  -m {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1,HcHcHc-3lv,HgHgHg-3lv}, --mode {isoHc-l1,isoHc-l2,isoHg-l2,isoHg-l1,HcHcHc-3lv,HgHgHg-3lv}
                        default is 'isoHc': use Hc method 
                        'isoHc-l1': Hc method with l1 postprocessing
                        'isoHc-l2': Hc method with l2 postprocessing
                        'isoHg-l2': Hg method with l2 postprocessing 
                        'isoHg-l1': Hg method with l1 postprocessing 
                        'HcHcHc-3lv': use Hc method with l1 postprocessing as estimate for consistency result 
                        'HgHgHg-3lv': use Hg method with l2 postprocessing as estimate for consistency result 
  -b BUDGET, --budget BUDGET
                        default is 1.0
  -d {taxi_lv1,haw_lv1}, --dataset {taxi_lv1,haw_lv1}
                        default is taxi dataset at top level

```

### Examlpes

To evaluate the Hc method (L1-isotonic regression) with budget 1.0 on taxi dataset (only the region of the root node):

```sh
$ python3 main.py -m isoHc-l1 -b 1.0 -d taxi_lv1
error:  2801.0
```


To get the consistency estimates using Hc method (L1 isotonic regression by default) with budget 1.0 on taxi dataset:

```sh
$ python3 main.py -m HcHcHc-3lv -b 1.0
```
The results are saved in files  "*./res/res_from_HcHcHc_r-{region_id}_w-weight_order-MSF_s-{seed}_b-{budget}_consistency*"

The error of the consistency result can be found in file "*./res/res_from_pds-taxi_prepr-HcHc_cds-taxi_lv3_crepr-Hc_w-weight_order-MSF_s-{seed}_b-{budget}_consistBatch.csv*"



To get the consistency estimates using Hg method (L2 isotonic regression by default) with budget 1.0 on taxi dataset:

```
$ python3 main.py -m HgHgHg-3lv -b 1.0
```
The results are saved in files  "*./res/res_from_HgHgHg_r-{region_id}_w-weight_order-MSF_s-{seed}_b-{budget}_consistency*"

The error of the consistency result can be found in file "*./res/res_from_pds-taxi_prepr-HgHg_cds-taxi_lv3_crepr-Hg_w-weight_order-MSF_s-{seed}_b-{budget}_consistBatch.csv*"

### To Use Your Data
non-hierarchical version: use *estimate_hc()* or *estimate_hg_l1()* or *estimate_hg_l2()* in `algs.py`.

For example:
```python
x_hat = estimate_hc(x, seed=20, cum_round=True, iso_norm='iso_l1', budget=budget)
```
hierarchical version: replace the retrieve_taxi_sizehist(dataset) in data.py 

## Versioning
* v0.2.0: features including hierarchical version of Hc and Hg methods
* v0.1.0: features including non-hierarchical version of Hc and Hg methods

## Citing
Please refer the paper:
```
@article{kuo11differentially,
      title={Differentially Private Hierarchical Count-of-Counts Histograms},
      author={Kuo, Yu-Hsuan and Chiu, Cho-Chun and Kifer, Daniel and Hay, Michael and Machanavajjhala, Ashwin},
      journal={Proceedings of the VLDB Endowment},
      volume={11},
      number={11}
}
```
