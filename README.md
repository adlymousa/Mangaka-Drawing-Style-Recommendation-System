Mangaka Drawing Style Identication
==================

Important Things
--------
- SlicingWithFaceDetectionAndTextExtraction.py is the last updated file to extract the features.

	input : The mangas from 109Manga data set
  
	It produces : 
  
		1- features.csv (List of features)
    
		2- mangaNames.csv (List of corresponding Manga names)
    
- MachineLearningReduced.py is the last updated file for machine learning.

	input : 
  
		1- features.csv (List of features)
    
		2- mangaNames.csv (List of corresponding Manga names)
    
	output : 
  
		- predicted.csv 
    
			C1 : Manga Name
      
			C2 : The first predicted Manga
      
			C3 : The Second predicted Manga
      
			C4 : The Third predicted Manga

- visualization.py is the last updated file for visualizing each feature.

	input : 
  
		1- features.csv (List of features)
    
		2- mangaNames.csv (List of corresponding Manga names)
    
	output : 
  
		- Plots, Change FeatureIndex (Value from 0 -> N) to plot each feature.
      
Current Workload
--------

	1- Checking each feature if needed.
	
	2- Eliminate overfetting.
	
	3- Writing the report.
