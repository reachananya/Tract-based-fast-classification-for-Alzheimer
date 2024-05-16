from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

subject_fol = ["003_S_4441_CN_F_69","003_S_4350_CN_M_73","109_S_4499_CN_M_84",
                  "003_S_4288_CN_F_73","003_S_4644_CN_F_68","007_S_4620_CN_M_77",
                  "098_S_4018_CN_M_76","098_S_4003_CN_F_78","094_S_4234_CN_M_70",
                  "021_S_4335_CN_F_73","021_S_4276_CN_F_75","016_S_2284_EMCI_M_73",
                  "016_S_4575_EMCI_F_62","016_S_2007_EMCI_F_84","005_S_4185_EMCI_M_81",
                  "007_S_2106_EMCI_M_81","005_S_2390_EMCI_F_89","016_S_4902_LMCI_F_77",
                  "016_S_4646_LMCI_F_62","016_S_4584_LMCI_F_78","007_S_4611_LMCI_M_68",
                  "003_S_4524_LMCI_M_72","003_S_4354_LMCI_M_76","021_S_4857_LMCI_M_68",
                  "003_S_2374_F_82_EMCI","057_S_4909_F_78_LMCI","003_S_4119_M_79_CN",
                  "057_S_4897_F_76_EMCI","094_S_2367_M_75_EMCI","003_S_4081_F_73_CN",
                  "094_S_2216_M_69_EMCI","094_S_4162_F_72_LMCI","094_S_4503_F_72_CN",
                  "094_S_4630_F_66_LMCI","098_S_0896_M_86_CN","098_S_2047_M_78_EMCI",
                  "094_S_4858_M_57_EMCI","098_S_2052_M_74_EMCI","094_S_4295_F_70_LMCI",
                  "094_S_4486_F_69_EMCI","094_S_4560_F_70_CN","098_S_2071_M_85_EMCI",
                  "098_S_2079_M_66_EMCI","098_S_4002_F_74_CN","098_S_4059_M_72_EMCI",
                  "027_S_4729_LMCI_F_78","021_S_4633_LMCI_F_73","021_S_4402_LMCI_F_73",
                  "005_S_4168_EMCI_M_82","007_S_4272_EMCI_M_72","016_S_2031_EMCI_M_73",
                  "016_S_4097_CN_F_71","007_S_4516_CN_M_72","007_S_4488_CN_M_73",
                  "005_S_0610_CN_M_89","003_S_4872_CN_F_69","003_S_4840_CN_M_62",
                  "003_S_4839_CN_M_66","003_S_4555_CN_F_66","005_S_4707_M_68_AD",
                  "005_S_5119_F_77_AD","003_S_4142_F_90_AD","003_S_5165_M_79_AD",
                  "003_S_4152_M_61_AD","005_S_5038_M_82_AD","005_S_4910_F_82_AD",
                  "003_S_4136_M_67_AD","003_S_4892_F_75_AD"]

print(len(subject_fol))

nbins = 5
nroi = 9

X_list = []
x_list_AD = []
labels_list = []

# Load the class matrices for each subject
for subject_id in subject_fol:
    class_mat = np.load(f'class_matrix_FA_21diff_{subject_id}.npy')
    class_matrix_AD = np.load(f'class_matrix_AD_21diff_{subject_id}.npy')
    X_list.append(class_mat)
    x_list_AD.append(class_matrix_AD)


    if "CN" in subject_id:
        label = 0
    else:
        label = 1

    labels_list.append(label)

# Combine features into one array
X1 = np.column_stack(X_list).T  #X - 70, 50 both FA + Ad - 70 + 100
X1[np.isnan(X1)] = 0
print(X1.shape)

X2 = np.column_stack(x_list_AD).T  #X - 70, 50 both FA + Ad - 70 + 100
X2[np.isnan(X2)] = 0
print(X2.shape)

X = np.hstack((X1, X2))
print(X.shape)

y = np.array(labels_list)
print(y.shape)

kf = KFold(n_splits=6)
kf.get_n_splits(X)
for i, (train_index, test_index) in enumerate(kf.split(X)):
  X_train = X[train_index,:]
  y_train = y[train_index]
  X_test = X[test_index,:]
  y_test = y[test_index]

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Get the feature importances
  feature_importances = rf_classifier.feature_importances_

  # Sort the features by importance
  sorted_indices = np.argsort(feature_importances)[::-1]

  # Print the top 10 most important features
  num_top_features = 10
  print("Top {} most important features:".format(num_top_features))
  for i in sorted_indices[:num_top_features]:
      print("Feature {}: {:.2f}".format(i, feature_importances[i]))
      region_num = ((i%[nbins*nroi]) // (nbins)) + 1
      print(region_num)

  # Train the random forest classifier
  rf_classifier = DecisionTreeClassifier(random_state=0)
  rf_classifier.fit(X_train, y_train)

  # Predict on the validation set
  y_pred = rf_classifier.predict(X_test)
  print(classification_report(y_test, y_pred))