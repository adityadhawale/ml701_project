# ml701_project

Google doc: https://docs.google.com/document/d/1kJw_3leF3Qn-W7OH7g-KyAmeOgMnthCGq2SCkoGy0As/edit

Dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101

Install vlfeat: http://www.vlfeat.org/install-matlab.html

## Sample Usage
`python3 phow_classifier.py --prefix logs/baseline- --classifier gnb --save_model True`

it should run and save a model. once you save the model, you just have to change the parameter from --save_model True to --pre_trained True

--classifier options are logistic_regression / gnb / svm

### Logistic regression 
`python phow_classifier.py --prefix logs/baseline- --classifier logistic_regression --classifier_args '{ "regularization": "l2", "solver": "lbfgs" }'`

`python phow_classifier.py --prefix logs/baseline- --classifier logistic_regression --classifier_args '{ "regularization": "l1", "solver": "saga", "max_iter": 10000, "warm_start": "True" }'`
