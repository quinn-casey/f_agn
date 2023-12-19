# f_agn
An empirical model to calculate the fractional AGN contribution in BPT-space. The model uses a Balmer decrement of Ha/Hb=2.86 to evolve 
emission line ratios (NII/Ha) & (OIII/Hb) from some starting point (starforming galaxy) to some ending point (AGN/LINER galaxy). We 
define 2500 starting/ending points  with the SDSS eBOSS DR16 data and  randomly connect points via emission line ratios and a 
fractional AGN contribution (0 < f_agn < 100). See Jones et al. 2016 (doi:10.3847/0004-637X/826/1/12) for a more complete explanation.

Usage:

`git clone https://github.com/quinn-casey/f_agn.git`

`from find_f import calc`
