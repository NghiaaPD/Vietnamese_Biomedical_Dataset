from transformers import pipeline

pipe = pipeline("text2text-generation", model="./models/Test_v1", device='cuda')

example = "Bệnh nhân nữ, 55 tuổi, tiền sử tiểu đường type 2, nhập viện do đau ngực kéo dài. Điện tâm đồ cho thấy dấu hiệu bất thường. Bệnh nhân được chỉ định chụp động mạch vành và kết quả cho thấy có hẹp động mạch 70%. Bác sĩ đề xuất đặt stent để cải thiện lưu thông máu. Ngoài ra, bệnh nhân có cholesterol cao và đang dùng atorvastatin."

summary = pipe(example, max_new_tokens=len(example))

print(summary)
