## Improvements
1. Without additional tuning, use open price as deal price can improve default qlib strategy. As open price is closer to the latest available pricing point, from 9.5% to 15.5%
2. Without additional tuning, adding benchmark value and the diff between stock and benchmark can provide some additional information. Such as if the price change is due to additional capital inflow, or due to capital chasing hot sector. Annualized exceed return improved from 9.5% to 12%
3. Without additional tuning, changing TopK Dropout trading strategy from topK=50, nDrop=5 to topK=50, nDrop=50 can increase the return wihtout cost, but it makes the return with cost has negative inforamtion ratio. This indicates the prediction is not good enough / provides enough alpha to cover transaction cost.
