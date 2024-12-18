# f_agn
An empirical model to calculate the fractional AGN contribution in BPT-space by decomposing the component parts (SFG & AGN). The model uses a Balmer decrement of Ha/Hb=2.86 to evolve 
emission line ratios (NII/Ha) & (OIII/Hb) from some starting point (starforming galaxy) to some ending point (AGN/LINER galaxy). We 
define 2500 starting/ending points with the SDSS eBOSS DR16 data and randomly connect points via emission line ratios and a 
fractional AGN contribution (0 < f_agn < 100). See Jones et al. 2016 (doi:10.3847/0004-637X/826/1/12) for a more complete explanation.

By default, the code uses the nearest 15 data points for the mean, median, standard deviation, and variance calculation. This can be altered by changing the optional `grid_name` parameter in the `calc()` function. Grids begining with 'r' use a radial selection (e.g. 'r025' takes all points within a 0.025 radius), and grids beginning with 'n' using the nearest n number of points (e.g. 'n5' grabs the nearest 5 points).

Disclaimer: Due to the nature of the model we do not distinguish between Seyferts and LINERs. Galaxies at low [NII]/Ha and high [OIII]/Hb are not well captured by this model.

Please include a footnote with the link to this repository if you use this model in your work.

Usage:

`git clone https://github.com/quinn-casey/f_agn.git`

`from find_f import calc`

`frac_agn = calc(x, y)`
