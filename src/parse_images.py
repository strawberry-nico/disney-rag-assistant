# src/parse_images.py
import os
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

# 你的 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def extract_table_from_image(image_path):
    print(f"🖼️ 正在解析图片: {image_path} ...")
    
    # 构造请求，让 Qwen-VL 模型看图说话
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"}, # 读取本地图片
                {"text": "请将这张图片中的表格完整提取为 Markdown 格式的文本。保留所有日期、房型和价格信息。"}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(
            model='qwen-vl-max', # 使用通义千问视觉大模型
            messages=messages
        )
        
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message.content[0]['text']
            print("✅ 解析成功！")
            return content
        else:
            print(f"❌ API 报错: {response.message}")
            return None
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return None

def main():
    # 1. 找图片
    image_dir = "raw_docs" # 假设你把 JPEG 放在这里
    output_dir = "processed_texts"
    
    # 支持的图片格式
    img_exts = ['.jpg', '.jpeg', '.png']
    
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in img_exts):
            img_path = os.path.join(image_dir, filename)
            
            # 2. 调用 AI 提取文字
            text_content = extract_table_from_image(img_path)
            
            if text_content:
                # 3. 保存为 txt
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                save_path = os.path.join(output_dir, txt_filename)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"💾 已保存到: {save_path}")

if __name__ == "__main__":
    main()