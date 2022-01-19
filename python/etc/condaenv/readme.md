This will create a conda env for fcdd named `fcdd_rc21`, where 'rc21' stands for "Reproducibility Challenge 2021".

After creating the environment, activate it, go to the fcdd/python and do `pip install --editable .`. 

Or run `fcdd_rc21.sh`.


---


Thalassa

I could not run the code on thalassa because the current CUDA version (10.1) is not enough for pytorch (>=10.2).

Downgrading pytorch

The max pytorch version with CUDA 10.1 is 1.8.1
src: https://pytorch.org/get-started/previous-versions/#v181

Use `fcdd_rc21_torch181.sh` instead.


