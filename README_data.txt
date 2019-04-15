Pipeline modifications:
	Amount of tiling for the spatial histograms
		hist_6
		hist_16
		hist-normal
	Number of words
		Words
			600 visual words
		Words-half
			300 visual words
		Words_twice
			1200 visual words
	Quantizer Methods (which method for quantizing into visual words)
		kd_tree
			Uses kd_tree method
		vq
	Featurization (three types of kernel that they can use)
		psix_kinters
		psix_kjs
		(default is chi2)
	Size of grid that makes dense keypoints
		hom_2
		hom_4
		hom_8
	Data augmentation (flipping, adding Gaussian noise)
		Transformed_also-hists
			Images flipped horizontal, vertical, both
		Default
			None
TODO:
	SIFT vs. ORB descriptors

