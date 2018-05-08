import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import csv


def train(train_dir, model_save_path):

    """
    Trains the model with given images and labels

    [train_dir] : directory to the train images
    [model_save_path] : the location where the trained model clf file to be saved

    """

    X = []
    y = []

    # Loop through each training image for the current person
    for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image)

        if len(face_bounding_boxes) ==1:

            # Add face encoding for current image to the training set
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)


    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, model_path):
    """
    Recognizes faces in given image using a trained KNN classifier

    [X_img_path] : path to the image to be predicted
    [model_path] : pickled clf file that was created from training

    """

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img, 1 ,"cnn")


    # If no faces are found in the image, return the predicted face location 
    if len(X_face_locations) == 0:
        X_face_locations = [(77L, 221L, 263L, 35L)]

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Predict classes and remove classifications that aren't within the threshold
    return zip(knn_clf.predict(faces_encodings))


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("KNN Classifier is being trained")
    classifier = train("images-train", model_save_path="4780_trained_knn_model.clf", n_neighbors=1)
    print("Traing is done. You can use the trained model to predict ur test data now. ")

    result = [['image_label', 'celebrity_name']]
    count = 0
    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("images-val-pub"):
        count +=1
        if count % 100 == 0: print ("Processed {} images".format(count))
        full_file_path = os.path.join("images-val-pub", image_file)
        
        predictions = predict(full_file_path, "4780_trained_knn_model.clf")

        append_helper = [str(image_file), predictions[0][0]]
        result.append(append_helper)


    with open("result.csv", 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(result)
