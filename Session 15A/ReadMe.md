# Assignment 15 A

Data Statistics
1) Background Images	 : 102				
2) Foreground Images     : 200 ( Including Flipped Images)				
3) Masks				 : 200				
4) Overlaid Images(fg_bg): 408000			
5) Overlaid Mask Images  : 408000				
6) Depth Images			 : 408000				

Mean of Images:
(Yet to revalidate) 
1) Overlaid Images(fg_bg): [0.5076, 0.4881, 0.4571]  				           
2) Overlaid Mask Images  : [0.0415, 0.0415, 0.0415]				
3) Depth Images			 : [0.9999, 0.9999, 0.9999]			


SD of Images:
(Yet to revalidate) 
1) Overlaid Images(fg_bg): [0.4985, 0.4985, 0.5419]			
2) Overlaid Mask Images  : [0.0612, 0.0612, 0.0612]			
3) Depth Images			 : [0.0015, 0.0015, 0.0015]				
		

Background Images: Different background images of the empty locations are downloaded from the internet. And each images is cropped 224x224 shapes. In total we have 102 images. Cropping portion is done using python code.                               
Ex: Empty Airports, Beaches, Stadiums, shopping malls, lounges, historic locations, parks etc.,                  

<img src="RM_Images\bg\bg001.jpg" style="height: 50; width:50;"/><img src="RM_Images\bg\bg002.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\bg\bg003.jpg" style="height: 50; width:50;"/><img src="RM_Images\bg\bg004.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\bg\bg005.jpg" style="height: 50; width:50;"/>

Foreground Images: Different png images of various objects/animals are downloaded again from google. These images are resized to 50x50 using python        
Ex: Humans, dogs, cats, trolley, bikes, cars, camers etc.,               

<img src="RM_Images\fg\fg002.png" style="height: 50; width:50;"/><img src="RM_Images\fg\fg024.png" style="height: 50; width:50;"/>
<img src="RM_Images\fg\fg033.png" style="height: 50; width:50;"/><img src="RM_Images\fg\fg057.png" style="height: 50; width:50;"/>
<img src="RM_Images\fg\fg153.png" style="height: 50; width:50;"/>

Masks : Masks are the created by replacing the all the empty cells in foreground images with black pixel values and non empty cells replaced with white pixel value using python              

<img src="RM_Images\masks\mk001.png" style="height: 50; width:50;"/><img src="RM_Images\masks\mk005.png" style="height: 50; width:50;"/>
<img src="RM_Images\masks\mk020.png" style="height: 50; width:50;"/><img src="RM_Images\masks\mk041.png" style="height: 50; width:50;"/>
<img src="RM_Images\masks\mk054.png" style="height: 50; width:50;"/>

Overlaid Images: All the foreground images are randomly placed on the background images in 20 different locations. Using this we created 408000 images dataset.
Overlaid Masks: in the same locations as the overlaid images, corresponding mask images are placed on the black images of 224x224          
<img src="RM_Images\ol\bg001_fg001_04.jpg" style="height: 50; width:50;"/><img src="RM_Images\ol\bg001_fg002_20.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\ol\bg001_fg003_12.jpg" style="height: 50; width:50;"/><img src="RM_Images\ol\bg001_fg009_09.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\ol\bg001_fg010_15.jpg" style="height: 50; width:50;"/>                                  
                                   
<img src="RM_Images\oms\bg001_mk002_09.jpg" style="height: 50; width:50;"/><img src="RM_Images\oms\bg001_mk003_12.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\oms\bg001_mk008_18.jpg" style="height: 50; width:50;"/><img src="RM_Images\oms\bg001_mk009_13.jpg" style="height: 50; width:50;"/>
<img src="RM_Images\oms\bg001_mk013_15.jpg" style="height: 50; width:50;"/>
                                
Depth Images: Depth Images are predicted using the pytorch code in the below link. Was not able to use the repository shared by you as it was taking lot of time to do predictions. Hence took a pytorch code and using GPU was able to predict the Depth images fastly.  
<img src="RM_Images\deps\bg001_fg001_12.png" style="height: 50; width:50;"/><img src="RM_Images\deps\bg001_fg003_01.png" style="height: 50; width:50;"/><img src="RM_Images\deps\bg059_fg110_17.png" style="height: 50; width:50;"/><img src="RM_Images\deps\bg50_fg033_05.png" style="height: 50; width:50;"/><img src="RM_Images\deps\bg102_fg191_03.png" style="height: 50; width:50;"/>                            
Repo Link: https://github.com/jai2shan/Depth-Estimation-PyTorch              
Script used to predict depth: https://github.com/jai2shan/Depth-Estimation-PyTorch/blob/master/test_jay.py                       


## Folder Location:
1) Background Images	 : https://drive.google.com/drive/folders/1hW9F3Z8Tu59DTFw7NwZqUH3Kp9Kqq5EK?usp=sharing          
2) Foreground Images     : https://drive.google.com/drive/folders/1hW9F3Z8Tu59DTFw7NwZqUH3Kp9Kqq5EK?usp=sharing                 
3) Masks				 : https://drive.google.com/drive/folders/1hW9F3Z8Tu59DTFw7NwZqUH3Kp9Kqq5EK?usp=sharing               
4) Overlaid Images(fg_bg): https://drive.google.com/drive/folders/1hW9F3Z8Tu59DTFw7NwZqUH3Kp9Kqq5EK?usp=sharing                 
5) Overlaid Mask Images  : https://drive.google.com/drive/folders/1AyagZcVFeM4IQ7aIvuAjz70BHTJOA-89?usp=sharing                   
6) Depth Images			 : https://drive.google.com/drive/folders/1AyagZcVFeM4IQ7aIvuAjz70BHTJOA-89?usp=sharing                
