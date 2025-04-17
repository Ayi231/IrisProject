# IrisProject.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# loading
iris = sns.load_dataset('iris')
print(iris.info())
print(iris.describe())

# pairplot
sns.pairplot(iris, hue="species")
plt.suptitle("Iris Feature Pairplot by Species", y=1.02)
plt.show()

# preparing data
X = iris.drop(columns='species')
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=42)
# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# dimensional reduction for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure()
for species, marker, color in zip(iris['species'].unique(), ['o','s','^'], ['r','g','b']):
    idx = y_train==species
    plt.scatter(X_pca[idx,0], X_pca[idx,1], marker=marker, label=species)
plt.legend()
plt.title("PCA Projection of Iris (Train Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# cross-validating
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
print(f"5‑Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# solving
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Test Set")
plt.show()
