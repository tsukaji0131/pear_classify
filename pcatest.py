import time
import matplotlib.pyplot as plt
import cv2
import numpy 
import pandas 
import pyefd 
import optuna

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import pickle

class ratio:
  r1 = 0
  r2 = 0
  r3 = 0
  r4 = 0
  r5 = 0
  r6 = 0
  r7 = 0
  r8 = 0
  r9 = 0

class width:
  w0 = 0
  w1 = 0
  w2 = 0
  w3 = 0
  w4 = 0
  w5 = 0
  w6 = 0
  w7 = 0
  w8 = 0
  w9 = 0 
  w10 = 0

#################################################################
# aspect ratio
#################################################################
def calc_aspect_ratio(contour):
  ellipse = cv2.fitEllipse(contour)
  aspect = ellipse[1][0]/ellipse[1][1]
  return aspect

#################################################################
# width ratio
#################################################################

def calc_width_ratio(contour):
  
  xlist = contour[:,0,0]
  ylist = contour[:,0,1]
  width_list = []
  for i in range(min(ylist),max(ylist)):
    pts = numpy.where(contour[:,0,1]==i)[0]

    if (pts.size != 0):
      xmin = xlist[min(pts)]
      xmax = xlist[max(pts)]
      dist = abs(xmin-xmax)
      if(dist >= 200 or i>0 ):
        width_list.append([dist,i,xmin,xmax])
  width_list = numpy.array(width_list)  

  width.w0 = width_list[numpy.argmax(width_list[:,0],axis=0)]
  width.w10 = width_list[0]
  distance = abs(width.w0[1]-width.w10[1])
  for i in range(1,10):
    exec("width.w{}=width_list[numpy.where(width_list[:,1] == width.w0[1]-int(distance*({}/10)))[0][0]]".format(i,i))
  for i in range(1,10):
    exec("ratio.r{} = width.w{}[0]/width.w0[0]".format(i,i))

  return ratio

def efd(contour,order): 
  coefficient = pyefd.elliptic_fourier_descriptors(numpy.squeeze(contour), order=order, normalize=True)
  return coefficient.flatten()[3:]

#################################################################
# contour extraction
#################################################################
def contour_extraction(img_org):

    img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    img_blur = cv2.blur(img_hsv, (10,10))
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    kernel = numpy.ones((10,10),numpy.uint8)
    closing = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    #cv2.drawContours(img_org, max_cnt, -1, (0, 0, 255), 10)

    #重心に原点を移動
    M = cv2.moments(max_cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    max_cnt[:,:,0] -= cx
    max_cnt[:,:,1] -= cy

    return max_cnt

#################################################################
# main
#################################################################
PCA_MODEL_PATH = "shape_classify\\model\\pcamodel.sav"
image_List = []
target = []
contour_List = []
coefficient_List = []
width_ratio_List = []
aspect_ratio_List = []
order = 10

file_path = "D:\\Le_Lectier\\2021\\before_ripe\\"
#自宅PC
# D:\\vscodepy3\\Le_Lectier\\2021\\before_ripe\\
#研究室PC
# D:\\2022_resarch_Tsukahara\\Data\\pear_data\\2021\\before_ripe\\

csv_input = pandas.read_csv(filepath_or_buffer="dataset.csv", encoding="ms932", sep=",")
dataset = csv_input.values

for columns in dataset:
    image_List.append(cv2.imread(file_path+"{}.bmp".format(columns[0])))
    target.append(columns[1])

for i in range(0,len(image_List)):
  contour = contour_extraction(image_List[i])
  coefficient_List.append(efd(contour=contour,order=order))
  aspect_ratio_List.append([calc_aspect_ratio(contour=contour)])
  width_ratio = calc_width_ratio(contour=contour)
  temp = []
  for j in range(1,10):
    exec("temp.append(width_ratio.r{})".format(j))
  width_ratio_List.append(temp)
coefficient_List = numpy.array(coefficient_List)
width_ratio_List = numpy.array(width_ratio_List)
aspect_ratio_List = numpy.array(aspect_ratio_List)

coefficient_List = numpy.concatenate([coefficient_List, width_ratio_List], 1)
print(coefficient_List.shape)
#coefficient_List = numpy.concatenate([coefficient_List, aspect_ratio_List], 1)
#print(coefficient_List.shape)

#PCA
#with open(PCA_MODEL_PATH, "rb") as f:
#  pca = pickle.load(f)
pca = PCA(n_components=10)
pcamodel = pca.fit(coefficient_List)
pickle.dump(pcamodel, open("pcamodel2.sav", "wb"))
feature = pcamodel.transform(coefficient_List)



random_state = 1
clf = svm.SVC(random_state=random_state)
#clf = RandomForestClassifier(random_state=random_state)
#clf = MLPClassifier(max_iter=2500,random_state=random_state )
#clf = GradientBoostingClassifier(random_state=random_state)
#clf = DecisionTreeClassifier(,random_state=random_state)
#clf = LogisticRegression(random_state=random_state)

scoring = { "p": "precision_macro",
            "a": "accuracy",
            "r": "recall_macro",
            "f": "f1_macro",
          }
skf = StratifiedKFold(n_splits=6,shuffle=True)
scores = cross_validate(clf, feature, target, cv=skf, scoring=scoring)
print("\nscores")
print(pandas.DataFrame(scores))
print("\naverage scores")
print("precision:",numpy.mean(scores['test_p']))
print("accuracy: ",numpy.mean(scores['test_a']))
print("recall:   ",numpy.mean(scores['test_r']))
print("F1:       ",numpy.mean(scores['test_f']))




#test_size = 0.4
#seed = 2
#
#X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=test_size, random_state=seed)
#clf.fit(X_train, Y_train)
#
#pickle.dump(clf, open("svmmodel2.sav", "wb"))
#
#predict = clf.predict(X_test)
#true = numpy.array(Y_test)
#print(predict)
#print(true)
#print(classification_report(true, predict))
#print (pandas.DataFrame(confusion_matrix(Y_test, predict),
#                    columns=["short","wide","others"],
#                    index=["short","wide","others"]))

