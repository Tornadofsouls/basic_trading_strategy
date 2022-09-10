## Improvements

Following trials is based on default example workflow on qlib, trying to experiment the effectiveness of some factor.

### 1. Use open price as deal price.
Without additional tuning, can improve default qlib strategy. As open price is closer to the latest available pricing point.

Annualized_return with cost improved from 9.5% to 15.5%.

### 2. Adding benchmark value and the diff between stock and benchmark
Without additional tuning, can provide some additional information. Such as if the price change is due to additional capital inflow, or due to capital chasing hot sector. 

Annualized exceed return with cost improved from 9.5% to 12%

### 3. Changing TopK Dropout trading strategy so that n_drop = topK
Without additional tuning, changing TopK Dropout trading strategy from topK=50, nDrop=5 to topK=50, nDrop=50 can increase the return wihtout cost. But it makes the return with cost has negative inforamtion ratio:

| Metric            | Before   | Without Cost | With Cost |
|-------------------|----------|--------------|-----------|
| annualized_return | 0.131321 | 0.215849     | -0.011118 |
| information_ratio | 1.496443 | 2.543944     | -0.131063 |

This indicates the prediction is not good enough / provides enough alpha to cover transaction cost. For example, stock A and stock B might diff in score 0.001 and changed rank between days a lot, which generates a lot of transaction fee but little gain.


### Combining 1.2.3.
Use open price change as training label, use open price as exchange deal price. Add benchmark feature for training. Before tuning lgbm params:

| Metric            | Default  | topk=50 n_drop=5 without cost | topk=50 n_drop=5 with cost | n_drop=50 without cost | n_drop=50 with cost |
|-------------------|----------|-------------------------------|----------------------------|------------------------|---------------------|
| annualized_return | 0.131321 | 0.080368                      | 0.043739                   | 0.381549               | 0.114158            |
| information_ratio | 1.496443 | 1.012313                      | 0.551214                   | 4.478850               | 1.336372            |

With the lgbm params tuned using optuna, using different timeframe and l2 valid error as target:

params = {'bagging_fraction': 0.6915989094330355, 'bagging_freq': 2, 'feature_fraction': 0.8133115108574889, 
              'lambda_l1': 0.008391240881677772, 'lambda_l2': 70.02732307226233, 'learning_rate': 0.014424878298136292, 
              'min_data_in_leaf': 196, 'num_leaves': 837, "max_depth": 10, "num_boost_round": 147, "verbosity": -1}

| Metric                        | Default  | topk=50 n_drop=5 without cost | topk=50 n_drop=5 with cost | n_drop=50 without cost | n_drop=50 with cost |
|-------------------------------|----------|-------------------------------|----------------------------|------------------------|---------------------|
| fixed 147 epoch annualized_return | 0.131321 | 0.056349                      | 0.015109                   | 0.341092               | 0.089175            |
| fixed 147 epoch information_ratio | 1.496443 | 0.578441                      | 0.155173                   | 3.679739               | 0.961535            |
| early stop 687 annualized_return  | 0.131321 | 0.078864                      | 0.037974                   | 0.367172               | 0.107527            |
| early stop 687 information_ratio  | 1.496443 | 0.924411                      | 0.445331                   | 4.247130               | 1.241823            |

This shows:
1. The params from the default workflow has good generalization ability, using early stopping, it can be used for different feature set to provide a baseline performance.
2. When to stop training influence the result a lot, with different number of training data, the epoch will be different. We cannot simply using the same number of epoch as the hyperparams for final model.
3. When training with early stopping validation set, the performance doesn't diff a lot for different lgbm hyper params.
4. Without all new factors above, we get a model performs much better without transactional cost, but perform worse with transactional cost. This indicates we can do some more works on trading strategy to fully utilize the prediction result. This also explained why the group performance chart seems amazing but back test result is bad.


---

Markdown table generator: https://www.tablesgenerator.com/markdown_tables
