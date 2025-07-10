# EduAid - Predicting School Dropout Risk
from sklearn.tree import DecisionTreeClassifier

# Fake training data: [attendance_rate, family_income, parent_education]
X = [[0.9, 50000, 2], [0.6, 10000, 0], [0.75, 30000, 1], [0.4, 5000, 0]]
y = [0, 1, 0, 1]  # 0 = stays in school, 1 = at risk of dropout

model = DecisionTreeClassifier()
model.fit(X, y)

# Test prediction
test_student = [[0.5, 8000, 1]]
prediction = model.predict(test_student)

if prediction[0] == 1:
    print("🚨 Student is at risk of dropping out.")
else:
    print("✅ Student likely to stay in school.")
