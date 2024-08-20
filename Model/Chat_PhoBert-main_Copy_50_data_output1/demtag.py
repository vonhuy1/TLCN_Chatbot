import json

# Đọc nội dung từ tệp JSON
file_path = 'contents.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Kiểm tra xem tệp JSON có chứa key nào được gọi là "tags" không
if 'tags' in data:
    # Đếm số lượng tag
    num_tags = len(data['tags'])
    print(f'Số lượng tag trong tệp JSON là: {num_tags}')
else:
    print('Tệp JSON không chứa key "tags".')
