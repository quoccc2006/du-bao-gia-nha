import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- 1. Đang đọc dữ liệu từ thư mục data/ ---")
# Đọc file dữ liệu huấn luyện
df = pd.read_csv('data/train.csv')

print("--- 2. Chọn đặc trưng và xử lý dữ liệu ---")
# Chọn 3 đặc trưng đơn giản, dễ hiểu để dự báo giá
features = ['OverallQual', 'GrLivArea', 'YearBuilt']
target = 'SalePrice' # Đây là cột giá nhà chúng ta cần dự báo

# Lấy dữ liệu đầu vào (X) và đầu ra (y)
# fillna() dùng để điền các giá trị bị thiếu bằng số trung vị (median) để tránh lỗi
X = df[features].fillna(df[features].median())
y = df[target]

print("--- 3. Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 4. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính ---")
model = LinearRegression()
model.fit(X_train, y_train) # Học mối quan hệ giữa đặc trưng và giá nhà

print("--- 5. Dự đoán và đánh giá mô hình ---")
y_pred = model.predict(X_test) # Cho mô hình dự đoán trên tập kiểm tra

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Sai số căn bậc hai (càng nhỏ càng tốt)
r2 = r2_score(y_test, y_pred) # Độ phù hợp (càng gần 1 càng tốt)

print(f"\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"RMSE (Sai số dự báo): {rmse:.2f} USD")
print(f"R2 Score (Độ chính xác tương đối): {r2:.4f}")

print("\n=== HỆ SỐ CỦA MÔ HÌNH (Coefficients) ===")
for feature, coef in zip(features, model.coef_):
    print(f"- {feature}: {coef:.2f}")
print(f"- Hệ số chặn (Intercept): {model.intercept_:.2f}")

print("\n✅ Hoàn thành! Mô hình đã chạy thành công.")