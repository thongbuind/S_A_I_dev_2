# Build a Pre-trained Language Model (LM) from scratch with Transformer architecture

## Test pre-train

*Mô hình được đào tạo trên tập tiếng việt có vocab_size = 10000, tập data 8.5tr token. Đã trải qua 1 epoch pretrain và 1 epoch continued-pretrain.*

*Từ test 1 đến test 11 là kiểm tra kiến thức lịch sử, yêu cầu phan biệt tên của các vị hoàng đế.*

*Từ test 12 đến test 19 là kiểm tra đặt câu. Input sẽ là 1 từ bất kì, không yêu cầu tính chính xác về ngữ nghĩa của câu, chỉ xem mô hình đã học được cách tạo câu cơ bản hay chưa.*

### Test 1
**Req:** hoàng đế thứ hai của triều đại nhà lý là

**Res:** hoàng đế thứ hai của triều đại nhà lý là lý thái tông

---

### Test 2
**Req:** lý công uẩn có miếu hiệu là

**Res:** lý công uẩn có miếu hiệu là lý thái tổ

---

### Test 3
**Req:** lý phật mã là tên thật của hoàng đế

**Res:** lý phật mã là tên thật của hoàng đế lý thái tông

---

### Test 4
**Req:** lý thái tông là

**Res:** lý thái tông là miếu hiệu của vua lý phật mã

---

### Test 5
**Req:** thánh tông hoàng đế của triều đại nhà lý là

**Res:** thánh tông hoàng đế của triều đại nhà lý là lý nhật tôn

---

### Test 6
**Req:** trần thái tổ có huý là

**Res:** trần thái tổ có huý là trần thừa

---

### Test 7
**Req:** trần cảnh là hoàng đế

**Res:** trần cảnh là hoàng đế trần thái tông

---

### Test 8
**Req:** trần thánh tông là

**Res:** trần thánh tông là miếu hiệu của trần hoảng

---

### Test 9
**Req:** thái tổ cao hoàng đế là thuỵ hiệu của

**Res:** thái tổ cao hoàng đế là thuỵ hiệu của lê lợi

---

### Test 10
**Req:** lê bang cơ có miếu hiệu là

**Res:** lê bang cơ có miếu hiệu là lê nhân tông

---

### Test 11
**Req:** thánh tông thuần hoàng đế là

**Res:** thánh tông thuần hoàng đế là miếu hiệu của vua lê thánh tông

**Nhận xét**: Không đúng lắm: phải là "thuỵ" chứ không phải "miếu".

---

### Test 12
**Req:** nhà trần

**Res:** nhà trần là một triều đại nổi bật trong lịch sử việt nam , nổi bật với những cải cách quan trọng trong thời kỳ này .

---

### Test 13
**Req:** phở

**Res:** phở là một món ăn truyền thống của việt nam , thường được chế biến từ các nguyên liệu tươi ngon và dễ chịu .

---

### Test 14
**Req:** giải đua xe công thức 1

**Res:** giải đua xe công thức 1 thường được xem là một trong những giải đua xe danh giá nhất thế giới .

---

### Test 15
**Req:** max verstappen

**Res:** max verstappen đã có nhiều đóng góp cho sự phát triển của thể thao đua xe , từ sự nghiệp đua xe đến kỹ năng .

---

### Test 16
**Req:** leclerc

**Res:** leclerc là một trong những tay đua nổi tiếng nhất trong lịch sử công thức 1 .

---

### Test 17
**Req:** vào buổi tối

**Res:** vào buổi tối , tôi đã học được nhiều bài học quý giá từ những câu chuyện lịch sử và văn hóa của dân tộc .

---

### Test 18
**Req:** sáng hôm ấy

**Res:** sáng hôm ấy , tôi cảm thấy mình rất thích và cảm xúc .

---

### Test 19
**Req:** sau khi ăn xong, chúng tôi

**Res:** sau khi ăn xong , chúng tôi sẽ đi dạo trong một ngày dài , tôi sẽ đi dạo với những món ăn đặc sản khác .


