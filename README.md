# colormap-changer

**Change the colormap of an old plot.**

*Hypothetical situation:* Let's say you want to use someone's old figure for a review paper. Unfortunately, they made their plot with a rainbow colormap and don't have the underlying data anymore - because this was a while ago. Nowadays, journals don't like rainbow colormaps anymore. And for a good reason: they are [nowhere near perceptually uniform](https://www.nature.com/articles/s41467-020-19160-7)*. So the journal might ask you to change the colormap. Now, all you got to work with is some rainbow-coloured pixels. So, naturally, you go down a rabbit hole for a couple of hours, trying to re-map color values to something perceptually uniform. Your code is pretty inefficient and the results don't look perfect, but at least it does something. 

Below an example. In this case the graticule on the lower subplot kind of messes things up. Should work better on higher-resolution images.

![example plot](output_comparison.jpg)

*Crameri, F., Shephard, G.E. & Heron, P.J. The misuse of colour in science communication. Nat Commun 11, 5444 (2020). https://doi.org/10.1038/s41467-020-19160-7
