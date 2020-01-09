
def zip_prediction_with_classes(prediction, classes):
    return dict(zip(classes, prediction))

def rounded_predictions(predictions, classes, treshold=0.5):
    zipped = zip_prediction_with_classes(predictions, classes)
    cleared = dict((key,value) for key, value in zipped.items() if value >= treshold)
    return cleared





