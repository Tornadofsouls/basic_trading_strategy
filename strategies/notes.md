## Improvements
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

This indicates the prediction is not good enough / provides enough alpha to cover transaction cost.


### Combining 1.2.3.
Use open price change as training label, use open price as exchange deal price. Add benchmark feature for training. Before tuning lgbm params:

| Metric            | Default  | topk=50 n_drop=5 without cost | topk=50 n_drop=5 with cost | n_drop=50 without cost | n_drop=50 with cost |
|-------------------|----------|-------------------------------|----------------------------|------------------------|---------------------|
| annualized_return | 0.131321 | 0.080368                      | 0.043739                   | 0.381549               | 0.114158            |
| information_ratio | 1.496443 | 1.012313                      | 0.551214                   | 4.478850               | 1.336372            |


Markdown table generator: https://www.tablesgenerator.com/markdown_tables
