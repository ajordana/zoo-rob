# This is the trajectory optimization benchmark for sampling based methods. 
The algorithms used are based on Hydrax (https://github.com/vincekurtz/hydrax)

# Setup

create a new environement 
```bash
conda env create -f environment.yml
```

Enter the conda env:

```bash
conda activate hydrax
```


Install the package and dependencies:

```bash
conda install /external/hydrax[dev]
```

Caution:

This benchmark can running into determinism issues related to GPU when running CubeRotation and Humaniod tasks (They are contact rich).

This is the result I get when running cube rotation on a CPU:
  0%|          | 1/200 [00:41<2:18:06, 41.64s/it]
🔥 best index:34 
🔥 best cost:0.22673562169075012 
🔥 current cost:0.47808676958084106 
  1%|          | 2/200 [01:13<1:57:39, 35.65s/it]
🔥 best index:1347 
🔥 best cost:0.19268843531608582 
🔥 current cost:0.22673562169075012 
  2%|▏         | 3/200 [01:45<1:51:28, 33.95s/it]
🔥 best index:771 
🔥 best cost:0.17842090129852295 
🔥 current cost:0.19268843531608582 
  2%|▏         | 4/200 [02:16<1:47:47, 33.00s/it]
🔥 best index:1208 
🔥 best cost:0.1420821249485016 
🔥 current cost:0.17842090129852295 
  2%|▎         | 5/200 [02:47<1:44:57, 32.30s/it]
🔥 best index:2036 
🔥 best cost:0.13830512762069702 
🔥 current cost:0.1420821249485016 
  3%|▎         | 6/200 [03:19<1:43:37, 32.05s/it]
🔥 best index:621 
🔥 best cost:0.12935101985931396 
🔥 current cost:0.13830512762069702 
  4%|▎         | 7/200 [03:50<1:42:48, 31.96s/it]
🔥 best index:616 
🔥 best cost:0.1215243712067604 
🔥 current cost:0.12935101985931396 

It is deterministic on a CPU!

This is the reuslt I get when running cube rotation on a GPU:
  0%|          | 1/200 [00:20<1:07:08, 20.24s/it]



🔥 best index:1476 
🔥 best cost:0.23488985002040863 
🔥 current cost:0.3948213458061218 
  1%|          | 2/200 [00:21<29:00,  8.79s/it]  



🔥 best index:1207 
🔥 best cost:0.19495776295661926 
🔥 current cost:0.24675482511520386 
  2%|▏         | 3/200 [00:21<16:44,  5.10s/it]



🔥 best index:1062 
🔥 best cost:0.15366610884666443 
🔥 current cost:0.19485774636268616 
  2%|▏         | 4/200 [00:22<11:06,  3.40s/it]



🔥 best index:1505 
🔥 best cost:0.1442430317401886 
🔥 current cost:0.17104315757751465 
  2%|▎         | 5/200 [00:23<07:53,  2.43s/it]



🔥 best index:1212 
🔥 best cost:0.10796670615673065 
🔥 current cost:0.1534000188112259 
  3%|▎         | 6/200 [00:23<05:58,  1.85s/it]



🔥 best index:203 
🔥 best cost:0.08504022657871246 
🔥 current cost:0.15056414902210236 
  4%|▎         | 7/200 [00:24<04:44,  1.47s/it]



🔥 best index:406 
🔥 best cost:0.07610496878623962 
🔥 current cost:0.08876136690378189 

Not deterministic!

Cannot do this as well, this gives some random bug, the values are clearly off.
It looks like this flag is buggy. https://github.com/jax-ml/jax/issues?q=is%3Aissue%20state%3Aopen%20xla_gpu_deterministic_ops%3Dtrue

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_deterministic_ops=true '
    '--xla_gpu_autotune_level=0' 
)

