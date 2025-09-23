## A. Tổng quan

### Kích thước mô hình

- **Embedding**: vocab_size × d_model = 10,000 × 64 = **640,000**  
- **Attention (mỗi layer)**: 4 × d_model² = 4 × 64² = **16,384**  
- **Feed Forward (mỗi layer)**: 2 × d_model × ff_dim = 2 × 64 × 256 = **32,768**  
- **Tổng mỗi layer**: 16,384 + 32,768 = **49,152**  
- **3 layer**: 3 × 49,152 = **147,456**  
- **Output head**: d_model × vocab_size = 64 × 10,000 = **640,000**

**Tổng tham số:** 640,000 + 147,456 + 640,000 = **1,427,456**

### Cách xây dựng
- Thiết kế Rotary Positional Embedding (RoPE), Multi-heads Attention sử dụng RoPE, mô hình decoder-only.
- 

---

## Data

### Crawl data
- Từ 2 nguồn chính là wikipedia và huggingface

### Làm sạch
- Xoá name tag của các báo.
- Xoá những kí hiệu lạ, không phải bảng chữ cái latin

---

## Tokenizer & Encoder

**Đầu ra ở bước này sẽ là 1 embedding_vector có kích thước [batch_size, seq_len, d_model]**

---

## Mô hình

### Pos Embedding

### Multi-heads Attention
- Tách d_model thành nhiều head: $d_k = d_{model} / num\_heads$, lúc này input sẽ có kích thước là [batch_size, seq_len, num_heads, d_k], reshape lại thành [batch_size, num_heads, seq_len, d_k] để thuận tiện sau tính ma trận Attention.
- Công thức Attention: $Softmax(\frac{Q.K^T} {\sqrt{d_k}}).V$
**Các bước tính toán**
- Đầu tiên, khởi tạo các lớp Dense trọng số:
    $W_q$ [d_q, d_model]
    $W_k$ [d_k, d_model]
    $W_v$ [d_v, d_model]
    trong đó **d_q = d_k = d_v = d_model / num_heads**
- Nhân nó với Embedding vector, kết quả sẽ là 3 vector Q, K, V.
- Tích hợp RoPE (viết chi tiết hơn).
- Tính **score** bằng cách lấy Q nhân với K^T rồi chia cho căn bậc 2 của d_k.
- causal mask (viết chi tiết hơn).
- Truyền qua Softmax, rồi nhân với V.
- Reshape (viết chi tiết hơn).

### Decoder block

### Model (Decoder-only)

---

## Train

### Create dynamic batch

- Chia data thành các batch. Với mỗi batch, lấy độ dài của seq dài nhất làm độ dài chung của cả batch rồi tiến hành padding theo độ dài đó.

- Nhưng rồi 1 vấn đề xuất hiện, Dynamic padding sẽ làm xuất hiện nhiều shape khác nhau, khiến Tensorflow phải retrace nhiều lần, tạo ra nhiều graph mới, tốn ram. Idea là sẽ cố định một số mốc padding như 50,100,150,...

- ==> Bucketing Padding.

### Split train/validation/test set

- Chia dataset thành 3 bộ một cách ngẫu nhiên, theo tỉ lệ 8/1/1.

- Mục đích của việc chia sẽ là: Mô hình được huấn luyện trên tập train, sau đó sẽ được đánh giá trên tập validation, điều chỉnh loss. Cuối cùng sau khi quá trình huấn luyện kết thúc, mô hình sẽ được đánh giá qua tập test. Mục đích là để tránh overfiting khi mà mô hình không được nhìn trước bài kiểm tra cuối cùng.

### Optimizer

*Là thuật toán dùng để tối ưu hoá weights trong backpropagation.*

#### Các optimizer cơ bản

- Đầu tiên là Gradient Descent: $w -= learning\_rate*gradient(x)$. Dấu "-" nghĩa là weight mới sẽ cập nhật ngược hướng của đạo hàm, giúp nghiệm hội tụ về cực tiểu. Thuật toán này thì cơ bản, dễ hiểu nhưng nó lại tồn tại nhiều nhược điểm như phụ thuộc quá nhiều vào giá trị khởi tạo và l_r, nếu đồ thị hàm số phức tạp, nó khó có thể tìm ra được global minimum, dễ bị quanh quẩn tại local minimum.

- Để khắc phục nhược điểm mắc kẹt tại local minimum của GD thì ta sử dụng thêm **Momentum**:    
        $$v_t = β*v_{t-1} + (1-β)*gradient(x_t)$$    
        $$w_{t+1} = w_t - learning\_rate*v_t$$    
    trong đó β là hệ số momen [0;1) thường là 0.9

- RMSProp 

- Là viết tắt của Adaptive Moment Estimation (Ước lượng moment thích nghi), một thuật toán tối ưu hoá dựa trên gradient, kết hợp ưu điểm của Momentum và RMSProp:
    * 

### Quá trình train

---

## Predict

