import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Dùng để lưu mô hình

print("Bắt đầu quá trình huấn luyện mô hình...")

# 1. Tải dữ liệu
df = pd.read_csv('Students Social Media Addiction1.csv')

# 2. Định nghĩa đặc trưng (X) và mục tiêu (y)
features = [
    'Gender',
    'Academic_Level',
    'Mental_Health_Score',
    'Avg_Daily_Usage_Hours',
    'Most_Used_Platform',
    'Sleep_Hours_Per_Night',
    'Conflicts_Over_Social_Media'
]
target = 'Addicted_Score'

X = df[features]
y = df[target]

# 3. Định nghĩa các cột số và cột phân loại
numeric_features = [
    'Mental_Health_Score', 
    'Avg_Daily_Usage_Hours', 
    'Sleep_Hours_Per_Night', 
    'Conflicts_Over_Social_Media'
]
categorical_features = [
    'Gender', 
    'Academic_Level', 
    'Most_Used_Platform'
]

# 4. Tạo bộ tiền xử lý (Preprocessor)
numeric_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Định nghĩa mô hình (đã được tối ưu)
model = DecisionTreeRegressor(max_depth=3, random_state=42)

# 6. Tạo Pipeline hoàn chỉnh
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# 7. Huấn luyện mô hình trên TOÀN BỘ dữ liệu
pipeline.fit(X, y)

print("Huấn luyện hoàn tất.")

# 8. Lưu mô hình ra tệp
model_filename = 'model.joblib'
joblib.dump(pipeline, model_filename)

print(f"Đã lưu mô hình thành công vào tệp: {model_filename}")