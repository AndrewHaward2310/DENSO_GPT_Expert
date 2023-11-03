import streamlit as st
from pathlib import Path
from PIL import Image
import os
from typing import List

current_dir: Path = Path(__file__).parent if "__file__" in locals() else Path.cwd()
assets_dir: Path = current_dir / "assets"
img_dir: Path = assets_dir / "img"
def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')

def get_files_in_dir(path: Path) -> List[str]:
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files


# with st.sidebar:
#     image = Image.open(f"{img_dir}/{get_files_in_dir(img_dir)[3]}")
#     st.image(image, caption='', use_column_width=True)
#     ui_spacer(1)
#     st.caption('Welcome to the DENSO GPT Expert')


image = Image.open(f"{img_dir}/{get_files_in_dir(img_dir)[3]}")
st.image(image, caption='', use_column_width=False, width=300)


st.write("### 📝 DENSO GPT Expert - Giải pháp thông minh cho nhà máy")
st.write("DENSO GPT Expert là sản phẩm demo được phát triển bởi Đội thi SmartABI tham gia cuộc thi Factory Hacks 2023 tổ chức bởi FPTxDENSO. Sản phẩm này là một AI Chatbot với chức năng thu thập tài liệu nhà máy và hỗ trợ nhân viên nhà máy trong việc xử lý các vấn đề liên quan dựa trên tài liệu thu thập được. Đội thi SmartABI cam kết mang đến những giải pháp thông minh và sáng tạo cho ngành công nghiệp.")

st.write("### 🤝 Member of SmartABI team:")
"1. [Lê Đức Nguyên](https://github.com/AndrewHaward2310)"
"2. [Trần Đức Đào Nguyên]()"
"3. [Trần Đăng An]()"
"4. [Nguyễn Xuân Trường](https://github.com/TruongNT95)"
"5. [Lữ Xuân Đức]()"


