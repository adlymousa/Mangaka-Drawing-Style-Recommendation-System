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

- Colleration.py is the last updated file for finding colleration for each feature.

	input : 
  
		1- features.csv (List of features)
    
		2- mangaNames.csv (List of corresponding Manga names)
    
	output : 
  
		- Collaration.csv (List of colleration value per feature).

		- CollarationMargin80%.csv (All features that have 80% colleration or more).
		
Current Workload
--------
	1- Writing the report.
	
	
Taken Features
--------
	1-  Feature_Page_Average_Area_To_Rectangle_Ratio.
	2-  Feature_Page_Average_Slope.
	3-  Feature_Page_Average_Canny_Lines.
	4-  Feature_SceneCanny_Average_Pixels.
	5-  Feature_Tone_Average_Pixels.
	6-  Feature_Outlines_To_Tones_Pixels_Ratio.
	7-  Density per unit area deviation.
	8-  Density per unit area.
	9-  FaceCanny (Vectors Pixels) Pixels per unit area.
	10- Feature_FaceTone_Pixels.
	11- Ratio FaceCanny / Face Tone.
	12- numberOfGradientColorsInFaces.
	13- numberOfGradientColorsInScenes.
