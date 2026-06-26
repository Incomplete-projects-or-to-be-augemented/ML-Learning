# Extreme Gradient Boosting

from prepData import Pipeline,XGBClassifier
from prepData import train_test_split,accuracy_score,confusion_matrix,classification_report
from prepData import x,y,preprocessor


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
