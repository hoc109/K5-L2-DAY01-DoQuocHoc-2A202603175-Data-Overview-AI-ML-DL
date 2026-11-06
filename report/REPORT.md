# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU (`Device: cpu`; không đổi model/checkpoint)

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145 (khớp bản ghim)

**Checkpoint:** `yolo11n-cls.pt` (sha256 `c62d41bf9625…`), `yolo11n.pt` (sha256 `0ebbc80d4a76…`), `yolo11n-seg.pt` (sha256 `55ed65c56c91…`) — khớp checksum ghim trong notebook

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên, không đổi)

**Mẫu ảnh:** `traffic` (COCO 210273, 640×428), `kitchen` (COCO 397133, 640×427), `dining` (COCO 166918, 480×640) — đều qua bước kiểm tra checksum của notebook

**Thay đổi so với notebook nguồn:** Không đổi code tạo prediction, checkpoint, threshold hay ảnh. Chỉ thêm **một ô phụ ở cuối** để nén thư mục `day1_lab_outputs/` thành ZIP và tải về máy (ô này không ảnh hưởng tới prediction). `REPORT.md` được viết trên máy và đưa thẳng vào `report/` của repo bằng Git.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K  
   (Các hạng tiếp theo: 2 `minibus` 0.164284, 3 `police_van` 0.085848, 4 `recreational_vehicle` 0.05411, 5 `streetcar` 0.048193.)

2. Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy`, `polygon_xy` hay `instance_id`, tức là không chỉ ra vị trí nào trong ảnh. Mỗi `sample_id` có đúng 5 record (rank 1–5), mỗi rank là một lớp được xếp hạng cho toàn ảnh. Trên PNG, ảnh `traffic` có hàng chục xe buýt, ô tô và người đi bộ, nhưng output vẫn chỉ trả **một** danh sách top-5 cho cả ảnh. Model không nói “chiếc xe nào” là `cab`.

3. Class list **đi kèm taxonomy/checkpoint** (ImageNet-1K, 1000 lớp cố định khi huấn luyện `yolo11n-cls.pt`), model không tự nghĩ ra. Model chỉ có thể chọn trong 1000 lớp đó. Bằng chứng: ImageNet-1K không có lớp “cảnh đường phố/giao thông”, nên model chọn lớp gần nhất là `cab`. Với ảnh `kitchen`, top-1 là `gong` (0.420192), nhiều khả năng vì những chảo đồng treo trên tường trông giống cồng, mà ImageNet lại không có lớp “nhà bếp”.

4. Phải ghi đủ cả ba trường vì:
   - `class_name` có thể trùng hoặc gần nghĩa giữa các taxonomy. Ví dụ `dining_table` (ImageNet, id 532) khác `dining table` (COCO-80, id 60).
   - `class_id` chỉ có nghĩa khi biết nó thuộc taxonomy nào. Id 468 trong ImageNet là `cab`, nhưng COCO-80 không có id 468.
   - `taxonomy_name` giúp truy vết và tái lập: biết nhãn sinh ra từ danh sách lớp nào, để map sang taxonomy của dự án mà không nhầm.

5. Nếu ảnh có nhiều vật/nhiều chủ đề, guideline cần nói rõ:
   - **Tiêu chí chọn lớp chính:** vật chiếm diện tích lớn nhất, vật ở trung tâm/tiêu điểm, hay chủ đề cảnh tổng thể. Ví dụ với `traffic` thì chọn “bus”, “car” hay “cảnh giao thông”?
   - **Thứ tự ưu tiên** khi hai lớp ngang nhau, kèm ví dụ ảnh minh họa đúng/sai.
   - **Dự án là single-label hay multi-label.** Nếu chỉ cho một nhãn mà ảnh thực sự có nhiều chủ đề thì có nhãn “khác/không xác định” không.
   - **Khi nào phải escalation:** không lớp nào trong taxonomy phù hợp, hoặc ảnh có vấn đề chất lượng. Ví dụ ảnh `traffic` có **viền đen trên và dưới** (thấy rõ trên PNG), cần quy định viền này có ảnh hưởng nhãn hay không.

6. Không lấy `score` làm ground truth vì score chỉ là mức “tin” của model, dùng để xếp hạng/lọc prediction. Nó không chứng minh prediction đúng. Score 0.51 của `cab` nghĩa là model chỉ hơi nghiêng về `cab`, còn gần một nửa xác suất rải sang các lớp khác. Ảnh `kitchen` cho thấy model có thể tự tin 0.42 với một nhãn sai rõ ràng (`gong`). Ground truth phải do người gán theo guideline rồi qua reviewer kiểm tra. Nếu lấy prediction làm nhãn, lỗi của model sẽ thành “sự thật” và bị học lại.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Record:** `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68  
   (`class_id`=0, `taxonomy_name`=COCO-80, `coordinate_unit`=pixel, `bbox_format`=xyxy, `score_threshold`=0.35, ảnh 640×427)

2. Gốc (0, 0) là góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới, đơn vị pixel.
   - **Góc trên-trái** của hộp ở (385.33, 69.24): cách mép trái khoảng 385 px, cách mép trên khoảng 69 px.
   - **Góc dưới-phải** ở (498.92, 348.92): cách mép trái khoảng 499 px, cách mép trên khoảng 349 px.
   - **Rộng** 498.92 − 385.33 = 113.58 px, **cao** 348.92 − 69.24 = 279.68 px.

   Hộp nằm ở nửa phải ảnh, cao khoảng 65% chiều cao ảnh, bao người đầu bếp đứng quay lưng, từ tóc xuống tới chân. Khớp với hộp `person 0.91` trên PNG.

3. Ảnh `kitchen`:
   - `0.20` → **17** prediction: 2 person, 5 bowl, 2 oven, 2 cup, 3 spoon, 1 potted plant, 1 dining table, 1 bottle
   - `0.35` → **11** prediction: 2 person, 5 bowl, 2 oven, 2 cup (đúng bằng số record `kitchen` trong JSON)
   - `0.60` → **6** prediction: person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61

   Ngưỡng càng cao thì càng ít hộp. Ở 0.60, các vật nhỏ/khó như 2 `cup` và 3 `bowl` có score thấp bị loại. Ở 0.20, model giữ thêm 6 hộp. Trong số đó có hộp đúng mà ở 0.35 bị bỏ sót (`dining table`, chiếc bàn gỗ lớn giữa ảnh), nhưng cũng có những hộp dễ sai lớp như `spoon` (muôi treo tường) và `potted plant` (bó rau khô treo).

4. Hạ ngưỡng thì giữ được nhiều hộp hơn, nhưng cũng thêm nhiều hộp sai. Reviewer phải:
   - Kiểm tra thêm nhiều hộp score thấp, loại các hộp sai lớp hoặc trùng lặp. Ví dụ trên PNG có hai hộp `cup` chồng lên nhau đến mức nhãn bị đè thành “cucup”, và hộp `bowl 0.38` [0.32, 305.02, 87.69, 371.32] chồng lên `bowl 0.72` và `bowl 0.50`.
   - Kiểm tra độ khít của hộp với các vật nhỏ, bị che hoặc cắt mép.
   - Vẫn phải soát lại vật bị bỏ sót. Threshold chỉ lọc prediction, không quyết định vật nào được bỏ qua khi tạo ground truth.

   Đổi lại, số vật bị bỏ sót sẽ giảm, nên cần cân bằng giữa thời gian review và độ phủ.

5. **Quy tắc box:**
   - Mỗi vật thể thuộc taxonomy nhìn thấy được phải có **một** hộp riêng (vật cùng lớp vẫn tách hộp).
   - Hộp chữ nhật phải **ôm khít phần nhìn thấy** của vật: cả bốn cạnh chạm điểm ngoài cùng của vật, không chừa nền thừa quá vài pixel và không cắt mất phần vật.
   - Vật bị che hoặc cắt mép chỉ vẽ phần nhìn thấy, không đoán phần bị che, và bật cờ `occluded`/`truncated` nếu dự án có.

   Quy tắc này không phụ thuộc vào score hay threshold của model.

6. **Vật bị cắt mép:** hộp `person` score 0.610931, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78], ở **mép trái** ảnh. Trên PNG chỉ thấy **một cánh tay/bàn tay** thò vào khung hình, phần còn lại của người nằm ngoài ảnh (x_min ≈ 0).
   - **Viết thành quy tắc guideline**, vì “vật bị cắt mép ảnh” lặp lại ở rất nhiều ảnh nên cần một rule chung: vẫn gán nhãn `person`, hộp ôm phần nhìn thấy, đánh dấu `truncated`.
   - **Escalation** chỉ khi guideline chưa định nghĩa ngưỡng phần nhìn thấy tối thiểu. Ví dụ: chỉ một ngón tay có tính là `person` không?
   - Ngược lại, những **nồi/chảo treo trên tường** không có hộp nào vì COCO-80 không có lớp “pot/pan”. Đây là câu hỏi về taxonomy (bỏ qua hay thêm lớp), nên **escalation** cho Lab Coach, annotator không được tự gán thành `bowl`.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Record:** `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]  
   (`bbox_xyxy`=[385.45, 66.44, 498.02, 348.58], `class_id`=0, `taxonomy_name`=COCO-80)

2. Box chỉ cho 4 số, tức một hình chữ nhật bao ngoài. Polygon 348 điểm cho **đường biên thật** của người: dáng đầu, vai, hai tay chống hông, tạp dề, khe giữa hai chân. Box gộp cả nền thừa trong hình chữ nhật, như góc khăn treo bên trái và chảo đồng bên phải, còn polygon loại phần nền đó ra.

   Tính diện tích polygon bằng công thức shoelace được khoảng **18 170 px²**, trong khi box rộng khoảng **31 760 px²**. Như vậy polygon chỉ chiếm khoảng **57%** box: hơn 40% diện tích box là nền. Với vật mảnh như `spoon` kitchen-008, tỉ lệ này chỉ khoảng 24%.

3. `instance_id` (dạng `<sample_id>-NNN`) là mã để **phân biệt từng instance trong bộ output của bài lab**. Ví dụ `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` đều là `bowl` (cùng `class_id` 45) nhưng là 4 cái bát khác nhau, mỗi cái có polygon riêng.
   - **Không phải class ID:** lớp được ghi ở `class_id`/`class_name`. `instance_id` khác nhau dù cùng lớp.
   - **Không phải tracking ID:** đây là ảnh tĩnh, mã chỉ là số thứ tự trong lần chạy này. Không có cam kết rằng `kitchen-001` ở ảnh/frame khác là cùng một người, và chạy lại với cấu hình khác thì thứ tự có thể đổi.

4. **Quy tắc vẽ biên mask:**
   - Mỗi instance có **một mask riêng**, đi sát mép vật trong khoảng 1–2 px.
   - Mask chỉ phủ **pixel nhìn thấy** của chính vật đó. Phần bị vật khác che ở phía trước phải **trừ ra**, kể cả vật đặt trên hoặc đứng trước.
   - Không phủ bóng đổ, phản chiếu hay nền lọt qua khe hở (khe giữa hai chân, quai cốc...).
   - Hai vật chạm nhau thì đường ranh giới đi theo cạnh nhìn thấy của vật nằm trước. Mask của hai instance không được chồng lên nhau.

5. **Chỗ hai vật chạm nhau / bị che:** mask `dining table` (kitchen-005, score 0.578416, 598 điểm, box [0.97, 234.74, 345.86, 423.87]) chạm và bị che bởi rất nhiều vật trên bàn: các `bowl` kitchen-002/003/006, lọ mứt, bột cán, khay bánh. Trên PNG, mask xanh còn **lan xuống cả chiếc xô kim loại dưới gầm bàn**, và mép trái chồng lấn với mask `oven`/`person` (kitchen-007, kitchen-009).
   - Phần “trừ vật che phía trước, không phủ vật khác” xử lý bằng **guideline**, theo quy tắc ở câu 4, vì ca này lặp lại ở mọi ảnh có vật đặt trên bàn.
   - Phần **phân loại** thì cần **escalation**:
     - Các khay/đĩa bánh nông có tính là `bowl` không? COCO-80 không có lớp “pie dish”.
     - Bó rau thơm khô treo trên tường đang bị đoán là `potted plant` (kitchen-004, 0.632235).
     - Muôi/vá treo tường đang bị đoán là `spoon` (kitchen-008, kitchen-011).

     Đây là câu hỏi taxonomy, annotator không tự quyết được.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một nhãn lớp cho cả ảnh** (single-label): `class_id` + `class_name` + `taxonomy_name`, không có tọa độ | `traffic` top-1 = `cab` 0.51 dù cảnh chủ yếu là xe buýt/ô tô. `kitchen` top-1 = `gong` 0.42, sai rõ vì ImageNet-1K không có lớp cảnh “street/kitchen”. Ảnh `traffic` có viền đen trên/dưới | Xem toàn ảnh, chọn **một** lớp theo tiêu chí lớp chính trong guideline, không chép top-1 của model. Không có lớp phù hợp thì escalation | Nhãn có đúng taxonomy/`class_id` không, có theo tiêu chí lớp chính không. Kiểm tra mẫu ngẫu nhiên các ảnh nhiều chủ đề. Ghi nhận ảnh lỗi chất lượng (viền đen) |
| Phát hiện vật thể | **Một hộp + một lớp cho mỗi vật**: `class_id`, `bbox_xyxy` pixel [x_min, y_min, x_max, y_max] | `kitchen`: hai hộp `cup` chồng nhau (“cucup”). `bowl 0.38` chồng `bowl 0.72`/`0.50`. `person 0.61` chỉ là cánh tay cắt mép trái. `oven 0.63` [489.09, 200.73, 617.81, 344.88] trông giống bồn rửa/tủ hơn. Nồi chảo treo không được phát hiện (ngoài COCO-80). `dining table` không có trong detection ở 0.35, chỉ xuất hiện ở 0.20 | Vẽ hộp khít cho **mọi** vật thuộc taxonomy, kể cả vật nhỏ/bị che/cắt mép (gắn cờ truncated). Không dựa vào threshold để bỏ vật | Đếm vật bị bỏ sót (ví dụ bàn), hộp trùng, sai lớp, hộp lỏng/hụt. Kiểm tra cờ truncated/occluded. Đối chiếu số hộp với ảnh |
| Instance segmentation | **Một polygon/mask + lớp cho mỗi instance**: `instance_id`, `class_id`, `polygon_xy` (điểm pixel bám biên) | `dining table` kitchen-005 lan xuống xô dưới gầm bàn, chồng lấn với `oven`/`person` bên trái. `potted plant` kitchen-004 thực ra là bó rau khô treo. `spoon` kitchen-008/011 là muôi treo tường | Vẽ mask riêng cho từng instance, sát biên, trừ phần bị che. Cùng lớp vẫn tách instance. Lớp mơ hồ thì escalation | Biên mask có sát không, có tràn sang vật khác hoặc chồng mask không, instance có bị gộp/tách sai không. Lớp có khớp taxonomy không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai đã kiểm tra checksum (giấy phép CC BY 2.0, ghi nguồn trong `IMAGE_ATTRIBUTION.md`). Không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai, và không ghi họ tên, MSSV, email, số điện thoại vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, ví dụ checksum sai, ảnh lạ hay dữ liệu cá nhân/nội bộ, tôi sẽ **dừng ngay**, không xử lý hoặc upload tiếp, và báo cho **Lab Coach/GV** qua kênh hỗ trợ của lớp. Nếu đã lỡ push, báo ngay để gỡ.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
