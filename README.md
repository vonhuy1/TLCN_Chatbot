# Đây là file readme hướng dẫn cài đặt và sử dụng chatbot
1. Tạo môi trường ảo `python -m venv venv`
2. Kích hoạt môi trường ảo `.\venv\Scripts\activate`
3. Tải các thư viện cần thiết theo `python3 -m pip install -r requirements.txt` 
hoặc `python -m pip install -r requirements.txt` đôi khi cú pháp là `pip install -r requirements.txt`
4. Lựa chọn môi trường ảo venv đã tạo và sử dụng.
5. Gõ vào trong cửa sổ Terminal của Visual Studio Code lệnh sau `streamlit run Home.py` hoặc `python -m streamlit run Home.py`
hoặc có thể là `python3 -m streamlit run Home.py` tùy theo phiên bản python sử dụng trên máy.(phiên bản em đang dùng là 3.11.2)(Lưu ý là dùng môi trường venv đã tạo.)
6. Lưu ý trước khi chạy phải mở một tab Google Chomre hoặc trình duyệt có chứa mail bất kỳ có số lượng cookies ít(Gmail trống và vô được trang web của Google Bard) để có thể sử dụng được Google Bard nếu dùng tài khoản mail chính khi chạy sẽ bị báo lỗi.

## Lưu ý
Lưu Ý: Khi sử dụng thanh cài đặt khác ví dụ như Rapid API,hugging Face thì phải tích 2 lần vào ô check box cùng tên ở dưới.
Tài khoản đăng nhập là: user và mật khẩu là: password
Các API key mẫu để Test khi dùng Chat GPT sử dụng key: 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
Với phần sử dụng davinci thì sẽ dùng:Key là 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
và Id tổ chức: org-MCbkAWezPoGDFyh6RYeE4mmk
## để sử dụng mô hình LLama
Mở terminal và chạy lệnh sau:
`python -m llama_cpp.server --model "./models/mistral-7b-openorca.Q4_0.gguf" --chat_format chatml --n_gpu_layers 1`



 