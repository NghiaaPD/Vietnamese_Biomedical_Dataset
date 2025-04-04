# Trích xuất thực thể có tên (Named Entity Recognition - NER) là gì?

NER (Named Entity Recognition) là một bài toán trong Xử lý ngôn ngữ tự nhiên (NLP), trong đó mô hình có nhiệm vụ xác định và phân loại các thực thể quan trọng trong văn bản vào các nhóm như tên bệnh, thuốc, triệu chứng, gen, bác sĩ, tổ chức y tế, v.v.

## 🔬 Ví dụ về NER trong y khoa

📄 Văn bản y khoa:

```
Bệnh nhân Nguyễn Văn A, 45 tuổi, được chẩn đoán mắc tiểu đường type 2 vào năm 2020. Anh ấy đang dùng Metformin để kiểm soát đường huyết. Bác sĩ Trần Bảo Minh khuyến nghị theo dõi biến chứng về thận.
```

📌 Kết quả trích xuất thực thể:

| Từ/ Cụm từ   |	Loại thực thể           |
|--------------|----------------------------|
| Nguyễn Văn A |	Bệnh nhân (Person)      |
| 45 tuổi      | 	Tuổi (Age)              |
| Tiểu đường type 2 |	Bệnh (Disease)      |
| Metformin	   | Thuốc (Drug)               |     
| Trần Bảo Minh |	Bác sĩ (Doctor)         |
| Thận         | Cơ quan cơ thể (Body Part) |

Mô hình sẽ gán nhãn các thực thể quan trọng để giúp các hệ thống trích xuất thông tin tự động từ hồ sơ y tế.

## 🩺 Ứng dụng của NER trong y khoa
- Tự động phân loại bệnh từ hồ sơ y tế.

- Trích xuất triệu chứng, loại thuốc và liều lượng từ đơn thuốc.

- Tóm tắt và phân tích dữ liệu y khoa từ tài liệu nghiên cứu.

- Hỗ trợ hệ thống hỏi đáp y tế (Medical QA).

- Hỗ trợ phân tích hồ sơ bệnh nhân trong bệnh viện.

## 📚 Dataset trong bài toán NER y khoa

### 1️. BC5CDR
Mô tả: Chứa hơn 1.500 bài báo PubMed được gán nhãn với bệnh (Disease) và hóa chất (Chemical).

Ứng dụng: Trích xuất tên thuốc, bệnh từ văn bản y khoa đã được tokenization.

URL: https://huggingface.co/datasets/tner/bc5cdr

#### Foramt data:

| tokens       |	tags                    |
|--------------|----------------------------|
| [ "Naloxone", "reverses", "the", "antihypertensive", "effect", "of", "clonidine", "." ] |	[ 1, 0, 0, 0, 0, 0, 1, 0 ] |

#### Label
```
{
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}
```

#### Data Splits

| Train | Validation | Test |
|-------|------------|------|
| 5228	| 5330       | 5865 |


### 2️. NCBI Disease Dataset
Mô tả: Gồm 693 bài báo từ PubMed với gán nhãn cho tên bệnh (Disease Names).

Ứng dụng: Phân loại và trích xuất tên bệnh từ văn bản.

#### Format data
```
{
  'tokens': ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous', 'polyposis', 'coli', 'tumour', 'suppressor', '.'],
  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0],
  'id': '0'
}
```
#### Data Fields
```
id: Sentence identifier.
tokens: Array of tokens composing a sentence.
ner_tags: Array of tags, where 0 indicates no disease mentioned, 1 signals the first token of a disease and 2 the subsequent disease tokens.
```

URL: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

### 3️. i2b2 2010 Challenge Dataset
Mô tả: Hồ sơ bệnh nhân được gán nhãn với bệnh, thuốc, bác sĩ, liều lượng thuốc.

Ứng dụng: Trích xuất thông tin từ hồ sơ bệnh nhân.

URL: https://www.i2b2.org/NLP/DataSets/Main.php

### 3️. ViMedNER

URL: https://github.com/tdtrinh11/ViMedNer/tree/main/data

## Model 

Model Link: 

# Bài toán Phân loại bệnh (Disease Classification) là gì?

Bài toán Phân loại bệnh (Disease Classification) trong NLP liên quan đến việc dự đoán loại bệnh từ văn bản y khoa như:

Hồ sơ bệnh án 🏥

Báo cáo xét nghiệm 🧪

Đơn thuốc 💊

Mô tả triệu chứng 🤒

Mô hình sẽ đọc văn bản đầu vào và dự đoán nhãn bệnh phù hợp dựa trên thông tin trong dữ liệu.

## 🔬 Ví dụ về phân loại bệnh

📄 Văn bản y khoa:

```
Tôi hiện đang có các triệu chứng như hay quên thông tin vừa tiếp nhận, khó tập trung và định hướng thời gian. Tôi có thể đang bị bệnh gì?
	
Tôi đang cảm thấy khó khăn khi tham gia cuộc hội thoại và hay lặp lại những câu chuyện đã kể. Tôi có thể đang bị bệnh gì?
	
Tôi hay quên nghĩa của từ và không thể theo dõi các cuộc trò chuyện. Tôi có thể đang bị bệnh gì?
	
Tôi hiện đang có các triệu chứng như thay đổi tâm trạng thất thường, dễ cáu gắt và lo lắng. Tôi có thể đang bị bệnh gì?
	
Tôi đang cảm thấy chán nản và muốn từ bỏ các sở thích, hoạt động xã hội. Tôi có thể đang bị bệnh gì?
	
Tôi hay quên mất ngày tháng, mùa và sự chuyển biến của thời gian. Tôi có thể đang bị bệnh gì?
```

📌 Kết quả dự đoán:

Alzheimer

## 📚 Dataset trong bài toán Disease Classification

### 1.Disease Diagnosis Dataset (Kaggle)
Mô tả: Chứa triệu chứng bệnh và chẩn đoán từ các bệnh viện.

Ứng dụng:

Dự đoán bệnh từ triệu chứng

Hỗ trợ chatbot y tế

Tải về: [Disease Diagnosis Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)



### 2.ViMedical_Disease

Tải về: [ViMedical_Disease](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)

### 3.ViMQ

Tải về: [ViMQ](https://github.com/tadeephuy/ViMQ/tree/main/data)

# Bài toán Tóm tắt hồ sơ y tế (Medical Text Summarization) là gì?

Bài toán Tóm tắt hồ sơ y tế (Medical Summarization) liên quan đến việc rút gọn thông tin y tế dài dòng thành bản tóm tắt ngắn gọn và súc tích, giúp bác sĩ hoặc bệnh nhân nhanh chóng hiểu nội dung chính.

## 🔬 Ví dụ về NER trong y khoa

📄  Đầu vào (Hồ sơ y tế gốc - dài):

```
"Bệnh nhân nữ, 55 tuổi, tiền sử tiểu đường type 2, nhập viện do đau ngực kéo dài. Điện tâm đồ cho thấy dấu hiệu bất thường. Bệnh nhân được chỉ định chụp động mạch vành và kết quả cho thấy có hẹp động mạch 70%. Bác sĩ đề xuất đặt stent để cải thiện lưu thông máu. Ngoài ra, bệnh nhân có cholesterol cao và đang dùng atorvastatin."
```

📌 Đầu ra (Bản tóm tắt - ngắn gọn):

```
'bệnh nhân nữ, 55 tuổi, tiền sử tiểu đường type 2, nhập viện do đau ngực kéo dài. Điện tâm đồ cho thấy hẹp động mạch 70%. bác sĩ đề xuất đặt stent để cải thiện lưu thông máu.'
```

## 🩺 Ứng dụng thực tế

✔ Hỗ trợ bác sĩ: Giúp bác sĩ xem nhanh thông tin chính mà không cần đọc toàn bộ hồ sơ.

✔ Hỗ trợ bệnh nhân: Cung cấp bản tóm tắt dễ hiểu từ ngôn ngữ chuyên ngành.

✔ Tự động hóa quy trình y tế: Rút gọn báo cáo bệnh án thành thông tin súc tích.

## 📚 Dataset cho bài toán Tóm tắt hồ sơ y tế
### 1.ViPubMed

Tải: https://huggingface.co/datasets/VietAI/vi_pubmed

### 1.VietMed-Sum

Tải: https://huggingface.co/datasets/leduckhai/VietMed-Sum

