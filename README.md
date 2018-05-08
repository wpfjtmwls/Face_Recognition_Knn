# Face_Recognition_Knn

## Citations

Our model uses two libraries to train and predict.

1. http://scikit-learn.org/stable/modules/neighbors.html
2. https://face-recognition.readthedocs.io/en/latest/face_recognition.html

## Installation

Install necessary packages by doing.

```bash
pip install -r requirements.txt
```

## Organization for inputs

The training/validation/testing image folder should be structured in the following way:

````bash
.
├── image_dir
│   ├── label (name of the person)
│   │   ├── image1
│   │   │   
│   │   └── image2
│   │      
````

## Output format

csv file where it is formatted as : 

````bash
image_label,celebrity_name
6d5e236555777bb62b4a251527977f80.jpg,saoirse_ronan
9ac2e0e97e2e06f32a01f242a3f71cc9.jpg,mila_kunis
c3d89497ba5973e592cf66e8816d066b.jpg,al_gore
....
````

## Future tweaks

For further uses, you can tweak the parameters under the main function. 
For example, you can train classifier on your own models by changing the train_dir and save_path and setting n_neighbors in STEP 1. 

````bash
# STEP 1: Train the KNN classifier and save it to disk
# Once the model is trained and saved, you can skip this step next time.
print("KNN Classifier is being trained")
classifier = train("images-train", model_save_path="4780_trained_knn_model.clf", n_neighbors=1)
print("Traing is done. You can use the trained model to predict ur test data now. ")
````

For prediction, you can change parameters for parameters for predict function and the location of validation/testing dir. The below part can be changed in STEP 2.

````bash
# STEP 2: Using the trained classifier, make predictions for unknown images
for image_file in os.listdir("images-val-pub"):
    count +=1
    if count % 100 == 0: print ("Processed {} images".format(count))
    full_file_path = os.path.join("images-val-pub", image_file)
    
    predictions = predict(full_file_path, "4780_trained_knn_model.clf")
````




