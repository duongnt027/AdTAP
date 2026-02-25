# Nghiên cứu và ứng dụng thuật toán Học tăng cường và Generative AI trong tăng cường dữ liệu ảnh

## Mô tả

Dự án tập trung nghiên cứu và ứng dụng các thuật toán Học tăng cường
(Reinforcement Learning) kết hợp với Generative AI nhằm tăng cường dữ
liệu ảnh. Mục tiêu là sinh ra các mẫu dữ liệu mới trong không gian tiềm
ẩn (latent space), từ đó cải thiện khả năng phát hiện bất thường và hiệu
suất của các mô hình thị giác máy tính.

------------------------------------------------------------------------

## Dữ liệu sử dụng

-   **Chính thức:**
    -   BMAD
-   **Ngoài ra (đang thử nghiệm):**
    -   PKU
    -   MV-Tec
    -   VisAD

------------------------------------------------------------------------

## Kỹ thuật sử dụng

### Generative Model

-   Sử dụng mô hình AutoEncoder để học biểu diễn tiềm ẩn `z` và tái tạo
    dữ liệu ảnh từ vector tiềm ẩn này.

### Reinforcement Learning

-   Sử dụng thuật toán PPO (Proximal Policy Optimization) để học cách
    sinh ra một nhiễu `delta z`.
-   Tạo vector tiềm ẩn mới:

```
z' = z + Δz
```
    

-   Vector `z'` được sử dụng để sinh ra dữ liệu ảnh mới thông qua
    decoder.

### Anomaly Detection

-   Sử dụng mô hình Transformer để phân biệt dữ liệu bình thường và dữ
    liệu bất thường.

------------------------------------------------------------------------

## Môi trường

-   Python 3.11
-   Các thư viện được liệt kê trong file `requirements.txt`
-   Các kết quả và notebook đã chạy được lưu trong thư mục:

```
results-ipynb/
```
    

------------------------------------------------------------------------

## Hướng dẫn chạy chương trình

### 1. Tải bộ dữ liệu

Download bộ dữ liệu BMAD từ nguồn:

https://github.com/DorisBao/BMAD

------------------------------------------------------------------------

### 2. Tiền xử lý dữ liệu

-   Tiền xử lý dữ liệu ảnh
-   Chia dữ liệu thành:
    -   train
    -   validation
-   Lưu dưới dạng `.npz` với:

```
allow_pickle=True
```
    

------------------------------------------------------------------------

### 3. Cấu hình đường dẫn

Thay đổi đường dẫn dữ liệu trong các file `main`.

------------------------------------------------------------------------

### 4. Huấn luyện mô hình

Chạy:

    python main.py

------------------------------------------------------------------------

## Pipeline tổng thể

    Image → Encoder → z → PPO → Δz → z' → Decoder → Generated Image
                                          ↓
                                     Transformer
                                          ↓
                                  Anomaly Detection
