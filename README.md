# f_agn
An empirical model to calculate the fractional AGN contribution in BPT-space by decomposing the component parts (SFG & AGN).
The model evolves emission line ratios ([NII]/Ha) & ([OIII]/Hb) from some starting point (starforming galaxy) to some ending point (AGN/LINER galaxy).
We assuming a Balmer decrement associated with case B recombination (e.g., 2.86 for starforming galaxies and 3.1 for AGN to account for collisional exitation). 
We define ~25000 starting/ending points based on the 2D distribution of SDSS eBOSS DR17 SFGs/AGN and randomly connect points via emission line ratios and a fractional AGN contribution (0 < f_agn < 100).

Results are stored in lookup tables for significantly faster computing.
We create and save 4 lookup tables of varying radii (r=[0.025, 0.05, 0.075, 0.1]) which you can switch between using the optional `grid_name` argument in the `calc()` function.
If no evolutionary tracks are contained within the radius then we select the 20 nearest points to estimate the fractional AGN contribution. 

Disclaimer: Due to the nature of the model we do not distinguish between Seyferts and LINERs.
Galaxies at low [NII]/Ha and high [OIII]/Hb are not well captured by this model (nor understood well physically -- I'm working on another paper to address this). 
You likely do not want to run `generate_table.py` locally as it is parallelized over 10 CPUs and takes a long time. 

Please include a footnote with the link to this repository if you use this model in your work. The associated paper is submitted to ApJ.

Usage:

`git clone https://github.com/quinn-casey/f_agn.git`

`from find_f import calc`

`frac_agn = calc(x, y)`
