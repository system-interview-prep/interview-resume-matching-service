FROM python:3.10

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y build-essential

# Tạo thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn vào container
COPY . /app

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy file .env nếu cần
# COPY .env /app/.env

# Mở port nếu chạy Flask
EXPOSE 5001

# Lệnh chạy ứng dụng
CMD ["python", "app.py"]
