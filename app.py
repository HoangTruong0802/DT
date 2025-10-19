import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Tên tệp dữ liệu mới
DATA_FILE = "teen_phone_addiction_dataset.csv"

# --- Hàm Huấn luyện Mô hình ---
@st.cache_resource
def get_model(file_path):
    """
    Hàm này tải dữ liệu, CHIA TÁCH, tiền xử lý, huấn luyện
    và CHẤM ĐIỂM mô hình.
    Nó cũng trả về các giá trị cho UI (dropdowns và sliders).
    """
    # 1. Tải dữ liệu
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy tệp dữ liệu '{file_path}'.")
        st.error("Vui lòng đảm bảo tệp CSV nằm cùng thư mục với tệp app.py.")
        return None, None, None, None, None, None

    # 2. Xác định đặc trưng và mục tiêu
    target_column = 'Addiction_Level'
    
    # Chọn các đặc trưng mới dựa trên dữ liệu
    features = [
        'Age',
        'Gender',
        'School_Grade',
        'Daily_Usage_Hours',
        'Sleep_Hours',
        'Academic_Performance',
        'Anxiety_Level',
        'Depression_Level',
        'Self_Esteem',
        'Phone_Usage_Purpose'
    ]
    
    X_all = df[features]
    y_all = df[target_column]

    # Phân loại đặc trưng
    numerical_features = [
        'Age', 
        'Daily_Usage_Hours', 
        'Sleep_Hours', 
        'Academic_Performance', 
        'Anxiety_Level', 
        'Depression_Level', 
        'Self_Esteem'
    ]
    categorical_features = [
        'Gender', 
        'School_Grade', 
        'Phone_Usage_Purpose'
    ]

    # 3. Chia dữ liệu: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # 4. Tạo bộ tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 5. Tạo Pipeline với mô hình đã "khử nhiễu"
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=7,
        min_samples_leaf=10,
        min_samples_split=20
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # 6. Huấn luyện mô hình CHỈ trên 80% dữ liệu (tập Train)
    pipeline.fit(X_train, y_train)

    # 7. Chấm điểm mô hình trên 20% dữ liệu lạ (tập Test)
    y_pred = pipeline.predict(X_test)
    model_score = r2_score(y_test, y_pred)

    # 8. Lấy các giá trị UI (cho selectbox và slider)
    unique_genders = df['Gender'].unique()
    unique_grades = sorted(df['School_Grade'].unique())
    unique_purposes = df['Phone_Usage_Purpose'].unique()
    
    # Tạo một dict chứa min/max cho các thanh trượt
    slider_ranges = {
        'Age': (int(df['Age'].min()), int(df['Age'].max())),
        'Daily_Usage_Hours': (0.0, 12.0), # Giữ cố định để dễ nhập
        'Sleep_Hours': (3.0, 10.0), # Giữ cố định
        'Academic_Performance': (int(df['Academic_Performance'].min()), int(df['Academic_Performance'].max())),
        'Anxiety_Level': (int(df['Anxiety_Level'].min()), int(df['Anxiety_Level'].max())),
        'Depression_Level': (int(df['Depression_Level'].min()), int(df['Depression_Level'].max())),
        'Self_Esteem': (int(df['Self_Esteem'].min()), int(df['Self_Esteem'].max()))
    }

    return pipeline, unique_genders, unique_grades, unique_purposes, slider_ranges, model_score

# --- Tải mô hình ---
try:
    (
        pipeline, 
        unique_genders, 
        unique_grades, 
        unique_purposes, 
        slider_ranges, 
        model_score
    ) = get_model(DATA_FILE)
    
    model_loaded = (pipeline is not None)
    
except Exception as e:
    st.error(f"Lỗi khi tải hoặc huấn luyện mô hình: {e}")
    model_loaded = False


# --- BẮT ĐẦU XÂY DỰNG GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Dự đoán Nghiện Điện thoại", layout="wide")
st.title("📱 Demo Mô hình Dự đoán Mức độ Nghiện Điện thoại (Teen)")
st.write("Nhập thông tin của học sinh vào thanh bên trái để mô hình dự đoán điểm nghiện (`Addiction_Level`).")
st.write("---")

if model_loaded:
    # --- Hiển thị điểm số của mô hình ---
    st.subheader("Đánh giá độ ổn định của mô hình")
    st.metric(label="Điểm tin cậy R-squared (trên dữ liệu Test)", value=f"{model_score:.4f}")
    if model_score < 0.3:
        st.error("Điểm quá thấp! Mô hình này không đáng tin cậy.")
    elif model_score < 0.6:
        st.warning(f"Điểm trung bình ({model_score:.1%}). Mô hình chỉ giải thích được một phần nhỏ.")
    else:
        st.success(f"Điểm khá tốt ({model_score:.1%})! Mô hình giải thích được phần lớn dữ liệu.")
    st.caption("Điểm $R^2$ (từ -∞ đến 1.0) đo lường mức độ mô hình dự đoán tốt trên *dữ liệu lạ*. Càng gần 1.0 càng tốt.")
    st.write("---")

    # --- Thanh bên (Sidebar) để nhập liệu ---
    st.sidebar.header("Nhập thông tin học sinh:")

    # --- Nhóm đặc trưng nhân khẩu học ---
    st.sidebar.subheader("Thông tin cơ bản")
    age = st.sidebar.slider(
        "Tuổi (Age):",
        min_value=slider_ranges['Age'][0], 
        max_value=slider_ranges['Age'][1], 
        value=15
    )
    gender = st.sidebar.selectbox(
        "Giới tính (Gender):",
        unique_genders
    )
    school_grade = st.sidebar.selectbox(
        "Khối lớp (School_Grade):",
        unique_grades
    )
    
    # --- Nhóm đặc trưng sử dụng ---
    st.sidebar.subheader("Thói quen sử dụng")
    daily_usage = st.sidebar.slider(
        "Giờ dùng trung bình/ngày (Daily_Usage_Hours):",
        min_value=slider_ranges['Daily_Usage_Hours'][0], 
        max_value=slider_ranges['Daily_Usage_Hours'][1], 
        value=5.0, 
        step=0.1
    )
    sleep = st.sidebar.slider(
        "Giờ ngủ/đêm (Sleep_Hours):",
        min_value=slider_ranges['Sleep_Hours'][0], 
        max_value=slider_ranges['Sleep_Hours'][1], 
        value=7.0, 
        step=0.1
    )
    phone_purpose = st.sidebar.selectbox(
        "Mục đích dùng chính (Phone_Usage_Purpose):",
        unique_purposes
    )

    # --- Nhóm đặc trưng tâm lý / học vấn ---
    st.sidebar.subheader("Sức khỏe & Học tập (1-10)")
    academic = st.sidebar.slider(
        "Kết quả học tập (Academic_Performance 0-100):",
        min_value=slider_ranges['Academic_Performance'][0], 
        max_value=slider_ranges['Academic_Performance'][1], 
        value=75
    )
    anxiety = st.sidebar.slider(
        "Mức độ Lo âu (Anxiety_Level):",
        min_value=slider_ranges['Anxiety_Level'][0], 
        max_value=slider_ranges['Anxiety_Level'][1], 
        value=5
    )
    depression = st.sidebar.slider(
        "Mức độ Trầm cảm (Depression_Level):",
        min_value=slider_ranges['Depression_Level'][0], 
        max_value=slider_ranges['Depression_Level'][1], 
        value=5
    )
    self_esteem = st.sidebar.slider(
        "Lòng Tự trọng (Self_Esteem):",
        min_value=slider_ranges['Self_Esteem'][0], 
        max_value=slider_ranges['Self_Esteem'][1], 
        value=5
    )


    # --- Nút dự đoán ---
    if st.sidebar.button("Nhấn để Dự đoán"):

        # 1. Tạo DataFrame từ dữ liệu nhập vào
        # Tên cột PHẢI Y HỆT như trong danh sách 'features'
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'School_Grade': [school_grade],
            'Daily_Usage_Hours': [daily_usage],
            'Sleep_Hours': [sleep],
            'Academic_Performance': [academic],
            'Anxiety_Level': [anxiety],
            'Depression_Level': [depression],
            'Self_Esteem': [self_esteem],
            'Phone_Usage_Purpose': [phone_purpose]
        }
        input_df = pd.DataFrame(input_data)

        st.subheader("Thông tin bạn đã nhập:")
        st.dataframe(input_df) 

        # 2. Gọi pipeline để dự đoán
        prediction = pipeline.predict(input_df)
        predicted_score = prediction[0]

        # 3. Hiển thị kết quả
        st.subheader("Kết quả Dự đoán:")
        
        # Dùng thang điểm 1-10 cho dễ hiểu (giống như các thang đo tâm lý)
        st.metric(
            label="Điểm Nghiện Dự đoán (Addiction_Level)",
            value=f"{predicted_score:.4f}",
        )
        
        # Đánh giá nhanh mức độ (Giả sử thang điểm 1-10)
        if predicted_score >= 8.0:
            st.error("🚨 Mức độ nghiện dự đoán: Rất Cao")
        elif predicted_score >= 6.0:
            st.warning("⚠️ Mức độ nghiện dự đoán: Cao")
        elif predicted_score >= 4.0:
            st.info("ℹ️ Mức độ nghiện dự đoán: Trung bình")
        else:
            st.success("✅ Mức độ nghiện dự đoán: Thấp")

    else:
        st.info("👈 Nhập thông tin ở thanh bên trái và nhấn nút 'Nhấn để Dự đoán'.")
