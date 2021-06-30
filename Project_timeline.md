## What have we done

**first day:** basic exploration, clean up, removing na, 

**day two:** basic models: ResNet and own model. RMSE is still about 14 years

**day three:** attempts to refine the models with limited success
    attempt included: manual cropping each image (180 images) and running models with the cropped images,
    with upsampling. RMSE still is about 10 years.



**today is day four (30 October):** our plans are 

    Cropping: segmentation part, 
    Thomas: Multi-input model with functional API to add the additional features.
    Tomek: segmentation, U-net with functional API
        finds specific part in an image, two boarders line (between darker lines)
    aproaches: manual preprocessing, if it helps, why not: cropping & selecting --> to get pretrained weights....

tomek: should push his codes to gitlab
additional goals: saving model parameters in google drive


**30 Oktober - 1. November: **
Tomek: U-net is not doable without image masks. In other words, one needs to preprocess the images first manually. 
preprocessing will be anyhow needed, but I doubt that U-net is the way to go. I also started to work with OpenCV. To test different filters.
a key problem I found, is that the typical edge filter identifies the lines as edges which are orthogonal to the lines we are interested in. 
Plus, the tooth lines are disconnected... 

my idea for Monday would be to discuss a strategy for the inevitable manual image preprocessing:
* manually crop the images, 
* set all images horizontally 
* use OpenCV (dilate/erode) to connect the horizonal lines
 


I am not sure whether this works. but here is a link to my codes. let me know if you can access it.
https://colab.research.google.com/drive/1XmPJqFVC-VsBDgVDo-CvJRaLJ7OmSR73?usp=sharing


**3 November**

What have we done so far: prefiltering the images and ResNet or EfficientNetB7 with either original images or cropped images.

goals: 
1) image masking and segmentation -> here get in contact with Matteo and DJ, at some point also with Barry
2) compare the models between cropped image subsets and the original images
3) filtering saw marks (can we do better?)
4) class balancing
5) data augmentation
6) probability layers
7) mlflow
8) compare with accuracy and certainty of the measurements based on human coding





** 9 November **

with Badru and Albin we discussed to try the following (6 November):

1) vertical cropping of the image 
2) Image segmentation
3) Grad Cam
4) aggregating by taking the mean of age predictions over all images of each person


**what we have achieved by 9 November**

Thomas masked images and started to make an image segmentation analysis
Tomek had the intention to do Grad Cam, but he is not yet done. He needs to ask Albin for help. 

But Tomek did some thinking: 
aggregating by taking the mean is not a good idea because the resulting value will gravitate towards the center of the distribution (that is, the resulting values will range probably between 50 and 60)

Alternatively, we can try multiple input model similarly to what Thomas tried with tooth code and sex as additional arrays. But multiple input model can also include multiple images for each individual into the model. For example, we can combine 2 original images and 2 cropped images into the model to predict the age of an individual (instead of predicting age of a single person associated with a single image)

Another think to consider is to use categories instead of linear regression. This might help the model to learn faster, more efficiently, and it has been suggested to use ordinal scaling (ranking) in recent articles on the use of face images to estimate age. 



**** November 17 *****

so far: 

Grad Cam
we managed to perform a Grad Cam Analysis with the cropped images.
in general, it seems that in case of correctly classified images the model does indeed learn from the tooth cementum.
Whenever, the model is paying attention to other aspects images are misclassified.
With age categorized into 7 bins we get an accuracy score of around 0.52 percent.
it is also possible to apply ordinal ranking, but I don't think we will manage to perform this analysis.
strangely enough, with regard to image classification ML approache practitioners rarely seem to care about ordinal ranking distributions.
And so, no tutorials can be found in the internet on this method apart from one stackflow comment


Image Segmentation
Thomas did a great job in setting up the image segmentation analysis. 
I am however not sure whether image segmentation works as we hoped it would do.


Multiple Input Model: 
Tomek performed yesterday a first run with slicing the cropped images and performing a multiple input model.
the Multiple image input model as done by Thomas on the original images seem to be the best performming model so far.
The same model but based on the manually horizontally aligned cropped images could be promising, 
but I had then to fix some issues (train set size, balancing). When reruning the model,
I could not perform the analysis 
because Google did not allow me to use GPU. Without GPU model analysis crashes.









#### 18 November, we are presenting 
check our presentation: https://drive.google.com/file/d/1SY7Pp9ZUhu3iraCtmiD9_dpK2aAfRORl/view?usp=sharing










