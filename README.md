# f_agn
An empirical model to calculate the fractional AGN contribution in BPT-space by decomposing the component parts (SFG & AGN).
The model uses a Balmer decrement of Ha/Hb=2.86 to evolve emission line ratios (NII/Ha) & (OIII/Hb) from some starting point (starforming galaxy) to some ending point (AGN/LINER galaxy).
We define ~25000 starting/ending points based on the 2D distribution of SDSS eBOSS DR17 SFGs/AGN and randomly connect points via emission line ratios and a fractional AGN contribution (0 < f_agn < 100).
See Jones et al. 2016 (doi:10.3847/0004-637X/826/1/12) for a more complete explanation.

Results are stored in lookup tables for significantly faster computing.
We create and save 4 lookup tables of varying radii (r=[0.025, 0.05, 0.075, 0.1]) which you can switch between using the optional `grid_name` argument in the `calc()` function.
If no evolutionary tracks are contained within the radius then we select the 20 nearest points to estimate the fractional AGN contribution. 

Disclaimer: Due to the nature of the model we do not distinguish between Seyferts and LINERs.
Galaxies at low [NII]/Ha and high [OIII]/Hb are not well captured by this model (nor understood well physically -- I'm working on another paper to address this). 
You likely do not want to run `generate_table.py` locally as it is parallelized over 10 CPUs and takes a long time. 

Please include a footnote with the link to this repository if you use this model in your work.

Usage:

`git clone https://github.com/quinn-casey/f_agn.git`

`from find_f import calc`

`frac_agn = calc(x, y)`
