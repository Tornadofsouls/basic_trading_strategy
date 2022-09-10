## Improvements
### Use open price as deal price.
Without additional tuning, can improve default qlib strategy. As open price is closer to the latest available pricing point.

Annualized_return with cost improved from 9.5% to 15.5%.

### Adding benchmark value and the diff between stock and benchmark
Without additional tuning, can provide some additional information. Such as if the price change is due to additional capital inflow, or due to capital chasing hot sector. 

Annualized exceed return with cost improved from 9.5% to 12%

### Changing TopK Dropout trading strategy so that n_drop = topK
Without additional tuning, changing TopK Dropout trading strategy from topK=50, nDrop=5 to topK=50, nDrop=50 can increase the return wihtout cost. But it makes the return with cost has negative inforamtion ratio:

| Metric            | Before   | Without Cost | With Cost |
|-------------------|----------|--------------|-----------|
| annualized_return | 0.131321 | 0.215849     | -0.011118 |
| information_ratio | 1.496443 | 2.543944     | -0.131063 |

This indicates the prediction is not good enough / provides enough alpha to cover transaction cost.
